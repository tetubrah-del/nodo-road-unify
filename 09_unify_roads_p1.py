#!/usr/bin/env python
"""
Raw road_links を統合して road_links_unified に保存するためのユーティリティ。

CLI 例:
    python 09_unify_roads_p1.py --run-unify --db-url postgresql://user:pass@localhost:5432/db
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, execute_batch
from shapely import wkt
from shapely.geometry import LineString

SNAP_THRESHOLD_M = 5.0
UNIFY_METHOD = "p1_unify"
RAW_SOURCES = ["manual", "gps", "osm", "satellite", "drone"]


@dataclass
class RawLink:
    link_id: int
    geom: LineString
    width_m: Optional[float]
    slope_deg: Optional[float]
    curvature: Optional[float]
    visibility: Optional[float]
    ground_condition: Optional[int]
    danger_score: Optional[float]
    source: Optional[str]
    metadata: Optional[Dict[str, Any]]


@dataclass
class UnifiedLink:
    geom: LineString
    width_m: Optional[float]
    slope_deg: Optional[float]
    curvature: Optional[float]
    visibility: Optional[float]
    ground_condition: Optional[int]
    danger_score: Optional[float]
    sources: List[str]
    parent_link_ids: List[int]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-unify", action="store_true",
                        help="raw road_links を統合して road_links_unified に保存する")
    parser.add_argument("--db-url", type=str, help="PostgreSQL 接続 URL")
    return parser.parse_args()


def load_raw_links(db_url: str) -> List[RawLink]:
    """raw road_links を取得する。

    source が manual/gps/osm/satellite/drone のものを対象とする。
    """
    sql = """
        SELECT
            link_id,
            ST_AsText(geom) AS wkt,
            width_m,
            slope_deg,
            curvature,
            visibility,
            ground_condition,
            danger_score,
            source,
            metadata
        FROM road_links
        WHERE source = ANY(%s)
           OR metadata->>'source' = ANY(%s)
    """

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (RAW_SOURCES, RAW_SOURCES))
            rows = cur.fetchall()
    finally:
        conn.close()

    raw_links: List[RawLink] = []
    for row in rows:
        (
            link_id,
            geom_wkt,
            width_m,
            slope_deg,
            curvature,
            visibility,
            ground_condition,
            danger_score,
            source,
            metadata,
        ) = row
        try:
            geom = wkt.loads(geom_wkt)
        except Exception:
            # geometry が壊れている場合はスキップ
            continue

        raw_links.append(
            RawLink(
                link_id=link_id,
                geom=geom,
                width_m=width_m,
                slope_deg=slope_deg,
                curvature=curvature,
                visibility=visibility,
                ground_condition=ground_condition,
                danger_score=danger_score,
                source=source,
                metadata=metadata,
            )
        )

    print(f"[INFO] Loaded raw road_links: {len(raw_links)}")
    return raw_links


def unify_links(raw_links: List[RawLink]) -> List[UnifiedLink]:
    """P1 の統合ロジックで raw road_links を統合する。

    ここでは既存のステージ 1 ロジックを保ちながら、
    ソースと元 link_id を追跡するメタデータを付与する。
    """
    unified: List[UnifiedLink] = []

    for raw in raw_links:
        metadata: Dict[str, Any] = raw.metadata.copy() if raw.metadata else {}
        unify_log = metadata.get("unify_log") or []
        unify_log.append(
            {
                "method": UNIFY_METHOD,
                "snap_threshold_m": SNAP_THRESHOLD_M,
                "parent_link_ids": [raw.link_id],
                "sources": [raw.source] if raw.source else [],
            }
        )
        metadata["unify_log"] = unify_log

        unified.append(
            UnifiedLink(
                geom=raw.geom,
                width_m=raw.width_m,
                slope_deg=raw.slope_deg,
                curvature=raw.curvature,
                visibility=raw.visibility,
                ground_condition=raw.ground_condition,
                danger_score=raw.danger_score,
                sources=[raw.source] if raw.source else [],
                parent_link_ids=[raw.link_id],
                metadata=metadata,
            )
        )

    print(f"[INFO] Unified road_links: {len(unified)}")
    return unified


def save_unified_links(db_url: str, unified_links: List[UnifiedLink]) -> None:
    """統合結果を road_links_unified に保存する。"""
    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    create_sql = """
        CREATE TABLE IF NOT EXISTS road_links_unified (
            link_id SERIAL PRIMARY KEY,
            geom GEOMETRY(LineString, 4326) NOT NULL,
            width_m REAL,
            slope_deg REAL,
            curvature REAL,
            visibility REAL,
            ground_condition INTEGER,
            danger_score REAL,
            sources TEXT[],
            parent_link_ids INTEGER[],
            metadata JSONB
        );
    """

    insert_sql = """
        INSERT INTO road_links_unified (
            geom,
            width_m,
            slope_deg,
            curvature,
            visibility,
            ground_condition,
            danger_score,
            sources,
            parent_link_ids,
            metadata
        )
        VALUES (
            ST_SetSRID(ST_GeomFromText(%s), 4326),
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s
        )
    """

    try:
        with conn.cursor() as cur:
            cur.execute(create_sql)

            records = [
                (
                    link.geom.wkt,
                    link.width_m,
                    link.slope_deg,
                    link.curvature,
                    link.visibility,
                    link.ground_condition,
                    link.danger_score,
                    link.sources,
                    link.parent_link_ids,
                    Json(link.metadata),
                )
                for link in unified_links
            ]

            execute_batch(cur, insert_sql, records, page_size=500)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"[INFO] Saved unified road_links: {len(unified_links)}")


if __name__ == "__main__":
    args = parse_args()

    if args.run_unify:
        if not args.db_url:
            raise ValueError("--db-url is required when --run-unify is set")

        raw_links = load_raw_links(args.db_url)
        unified_links = unify_links(raw_links)
        save_unified_links(args.db_url, unified_links)
    else:
        print("[INFO] No action specified. Use --run-unify to execute the pipeline.")
