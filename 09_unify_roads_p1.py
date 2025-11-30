#!/usr/bin/env python
"""
PostGIS に格納された複数ソース(manual/gps/osm/satellite/drone)の road_links を
優先度と重み付けに従って統合し、スナップ/マージ/結合処理を行った上で
unified_road_links テーブルへ upsert するスクリプト。
"""

import argparse
import os
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_batch
from shapely.geometry import LineString
from shapely.ops import linemerge, snap, unary_union
from shapely import wkb


SOURCE_PRIORITY: Dict[str, int] = {
    "manual": 0,
    "gps": 1,
    "osm": 2,
    "satellite": 3,
    "drone": 4,
}

DEFAULT_SNAP_DISTANCES = (1.0, 3.0, 5.0)
MERGE_DISTANCE = 1.0
JOIN_DISTANCE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unify road links from multiple sources")
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("NODO_DB_URL") or os.environ.get("DATABASE_URL"),
        help="PostGIS connection string",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PRIORITY.keys()),
        help="Sources to unify",
    )
    parser.add_argument(
        "--snap-distances",
        nargs="+",
        type=float,
        default=list(DEFAULT_SNAP_DISTANCES),
        help="Snap distances (meters) applied in ascending order",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write unified data back to PostGIS",
    )
    return parser.parse_args()


# -----------------------------
# DB helpers
# -----------------------------

def fetch_source_links(conn, sources: Sequence[str]) -> List[Dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT link_id, source, confidence, metadata, parent_link_id, ST_AsEWKB(geometry)
            FROM road_links
            WHERE source = ANY(%s)
            """,
            (list(sources),),
        )
        rows = cur.fetchall()

    links: List[Dict] = []
    for link_id, source, confidence, metadata, parent_link_id, geom_wkb in rows:
        geom = wkb.loads(geom_wkb)
        links.append(
            {
                "link_id": link_id,
                "source": source,
                "confidence": confidence,
                "metadata": metadata or {},
                "parent_link_ids": [parent_link_id] if parent_link_id else [link_id],
                "geometry": geom,
            }
        )
    return links


def fetch_next_link_id(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(link_id), 0) FROM unified_road_links")
        (max_id,) = cur.fetchone()
    return int(max_id) + 1


# -----------------------------
# Geometry helpers
# -----------------------------

def snap_geometry(geom: LineString, anchors: Iterable[LineString], distances: Sequence[float]) -> LineString:
    union = unary_union(list(anchors)) if anchors else None
    snapped = geom
    for dist in distances:
        if union:
            snapped = snap(snapped, union, dist)
    return snapped


def merge_nearby_geometries(links: List[Dict], distance: float) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    remaining = links[:]
    while remaining:
        base = remaining.pop()
        merged = [base]
        base_geom = base["geometry"]
        keep: List[Dict] = []
        for candidate in remaining:
            if base_geom.distance(candidate["geometry"]) <= distance:
                merged.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        groups.append(merged)
    return groups


def join_disconnected_geometries(geom: LineString, distance: float) -> LineString:
    buffered = geom.buffer(distance, cap_style=2, join_style=2)
    merged = linemerge(buffered)
    if isinstance(merged, LineString):
        return merged
    # fall back to original when linemerge fails
    return geom


# -----------------------------
# Unification pipeline
# -----------------------------

def build_unified_links(
    links: List[Dict],
    snap_distances: Sequence[float],
) -> List[Dict]:
    links_sorted = sorted(links, key=lambda l: SOURCE_PRIORITY.get(l["source"], 100))
    anchors: List[LineString] = []
    snapped_links: List[Dict] = []

    for link in links_sorted:
        geom = snap_geometry(link["geometry"], anchors, snap_distances)
        geom = join_disconnected_geometries(geom, JOIN_DISTANCE)
        snapped = {**link, "geometry": geom}
        snapped_links.append(snapped)
        anchors.append(geom)

    merged_groups = merge_nearby_geometries(snapped_links, MERGE_DISTANCE)

    unified: List[Dict] = []
    for group in merged_groups:
        geoms = [g["geometry"] for g in group]
        union_geom = unary_union(geoms)
        merged_geom = linemerge(union_geom)
        if not isinstance(merged_geom, LineString):
            merged_geom = max(
                (geom for geom in geoms if isinstance(geom, LineString)),
                key=lambda g: g.length,
                default=LineString(),
            )

        parent_ids: List[int] = []
        metadata = defaultdict(list)
        for g in group:
            parent_ids.extend(g.get("parent_link_ids", []))
            metadata["sources"].append(g["source"])
            if g.get("metadata"):
                metadata["metadata"].append(g["metadata"])

        unified.append(
            {
                "geometry": merged_geom,
                "parent_link_ids": sorted(set(parent_ids)),
                "metadata": {"source": list(dict.fromkeys(metadata["sources"])), "children": metadata["metadata"]},
            }
        )

    return unified


# -----------------------------
# PostGIS writer
# -----------------------------

def upsert_unified_links(conn, unified_links: List[Dict]) -> None:
    next_id = fetch_next_link_id(conn)
    rows: List[Tuple[int, List[int], str, bytes]] = []
    for idx, link in enumerate(unified_links, start=next_id):
        metadata = link.get("metadata") or {}
        metadata["parent_link_ids"] = link.get("parent_link_ids", [])
        rows.append(
            (
                idx,
                link.get("parent_link_ids", []),
                json.dumps(metadata),
                link["geometry"].wkb,
            )
        )

    with conn.cursor() as cur:
        execute_batch(
            cur,
            """
            INSERT INTO unified_road_links (link_id, parent_link_ids, metadata, geometry)
            VALUES (%s, %s, %s, ST_SetSRID(%s::geometry, 4326))
            ON CONFLICT (link_id) DO UPDATE
            SET parent_link_ids = EXCLUDED.parent_link_ids,
                metadata = EXCLUDED.metadata,
                geometry = EXCLUDED.geometry
            """,
            rows,
        )
    conn.commit()


# -----------------------------
# main
# -----------------------------

def main():
    args = parse_args()
    if not args.db_url:
        raise SystemExit("Database URL is required")

    conn = psycopg2.connect(args.db_url)
    links = fetch_source_links(conn, args.sources)
    unified_links = build_unified_links(links, args.snap_distances)

    if args.dry_run:
        for link in unified_links[:5]:
            print("geometry length", link["geometry"].length)
            print("parent ids", link["parent_link_ids"])
            print("metadata", link["metadata"])
    else:
        upsert_unified_links(conn, unified_links)
        print(f"Inserted/updated {len(unified_links)} unified links")
    conn.close()


if __name__ == "__main__":
    main()
