#!/usr/bin/env python
"""
GPX ログ -> PostGIS(road_links) インポートスクリプト

使い方例:
  export NODO_DB_URL="postgresql://nodo:nodo_password@localhost:5432/nodo"

  python 07_import_gps_logs.py \
      --gpx-path data/gps/annoura.gpx \
      --source gps \
      --confidence 1.0
"""

import os
import argparse
import json
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_batch
import xml.etree.ElementTree as ET


GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import GPX tracks into PostGIS road_links"
    )
    parser.add_argument(
        "--gpx-path",
        type=str,
        required=True,
        help="GPX ファイルパス",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("NODO_DB_URL") or os.environ.get("DATABASE_URL"),
        help=(
            "Postgres connection URL "
            "(例: postgresql://nodo:nodo_password@localhost:5432/nodo). "
            "未指定なら NODO_DB_URL / DATABASE_URL を使う"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gps",
        help="road_links.source に入れる値 (default: gps)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="road_links.confidence に入れる値 (default: 1.0)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="このポイント数未満のトラックは捨てる (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB書き込みを行わず、読み取り結果のみ表示",
    )
    return parser.parse_args()


# -----------------------------
# GPX 解析
# -----------------------------

def parse_gpx_tracks(
    gpx_path: str,
    min_points: int = 2,
) -> List[List[Tuple[float, float]]]:
    """
    GPX ファイルからトラックを抽出し、各トラックを [(lon, lat), ...] のリストとして返す。

    - gpx:trk ごと
    - その中の gpx:trkseg ごとに一本のラインとして扱う
    """
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    tracks = []
    for trk in root.findall("gpx:trk", GPX_NS):
        for seg in trk.findall("gpx:trkseg", GPX_NS):
            coords: List[Tuple[float, float]] = []
            for pt in seg.findall("gpx:trkpt", GPX_NS):
                lat = float(pt.attrib["lat"])
                lon = float(pt.attrib["lon"])
                coords.append((lon, lat))

            if len(coords) >= min_points:
                tracks.append(coords)

    return tracks


# -----------------------------
# WKT 生成
# -----------------------------

def build_linestring_wkt(coords: List[Tuple[float, float]]) -> str:
    """
    [(lon, lat), ...] -> 'LINESTRING(lon lat, ...)' のWKTを生成
    """
    coord_str = ", ".join(f"{lon} {lat}" for lon, lat in coords)
    return f"LINESTRING({coord_str})"


# -----------------------------
# PostGIS 書き込み
# -----------------------------

def insert_gps_road_links(
    db_url: str,
    tracks: List[List[Tuple[float, float]]],
    gpx_path: str,
    source: str,
    confidence: float,
):
    """
    tracks を road_links に一括インサートする。

    - geom: ST_GeomFromText(..., 4326)
    - width_m, slope_deg, curvature, visibility, ground_condition, danger_score, parent_link_id は NULL
    - metadata に gpx_path と track_index を保存
    """
    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            records = []
            for idx, coords in enumerate(tracks):
                if len(coords) < 2:
                    continue

                wkt = build_linestring_wkt(coords)

                metadata = {
                    "source_gpx": os.path.basename(gpx_path),
                    "track_index": idx,
                    "note": "imported from GPX track",
                }
                metadata_json = json.dumps(metadata, ensure_ascii=False)

                # SQL のプレースホルダ順に合わせる: (wkt, metadata_json, source, confidence)
                records.append(
                    (
                        wkt,
                        metadata_json,
                        source,
                        confidence,
                    )
                )

            if not records:
                print("[WARN] No valid tracks to insert.")
                return

            sql = """
                INSERT INTO road_links (
                    geom,
                    width_m,
                    slope_deg,
                    curvature,
                    visibility,
                    ground_condition,
                    danger_score,
                    metadata,
                    source,
                    confidence,
                    parent_link_id
                )
                VALUES (
                    ST_SetSRID(ST_GeomFromText(%s), 4326),
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    %s::jsonb,
                    %s,
                    %s,
                    NULL
                )
            """
            execute_batch(cur, sql, records, page_size=100)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# -----------------------------
# main
# -----------------------------

def main():
    args = parse_args()

    if not args.db_url and not args.dry_run:
        raise RuntimeError(
            "DB URL が指定されていません。--db-url か NODO_DB_URL / DATABASE_URL を設定してください。"
        )

    print(f"[INFO] Loading GPX from: {args.gpx_path}")
    tracks = parse_gpx_tracks(args.gpx_path, min_points=args.min_points)
    print(f"[INFO] Parsed {len(tracks)} track(s) from GPX.")

    if not tracks:
        print("[WARN] GPX から有効なトラックが見つかりませんでした。終了します。")
        return

    if args.dry_run:
        total_points = sum(len(t) for t in tracks)
        print(f"[DRY RUN] tracks={len(tracks)}, total_points={total_points}")
        return

    print(
        f"[INFO] Inserting tracks into road_links "
        f"(source={args.source}, confidence={args.confidence})..."
    )
    insert_gps_road_links(
        db_url=args.db_url,
        tracks=tracks,
        gpx_path=args.gpx_path,
        source=args.source,
        confidence=args.confidence,
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
