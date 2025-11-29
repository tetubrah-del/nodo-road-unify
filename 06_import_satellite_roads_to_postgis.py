#!/usr/bin/env python
"""
衛星スケルトンPNG -> PostGIS(road_links) インポートスクリプト

使い方例:
  export NODO_DB_URL="postgresql://user:pass@localhost:5432/nodo"
  python 06_import_satellite_roads_to_postgis.py \
      --skeleton-path road_line_mask_skel_v3.png \
      --min-lat 33.123456 \
      --max-lat 33.132456 \
      --min-lon 131.234567 \
      --max-lon 131.243567 \
      --min-points 10
"""

import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import psycopg2
import json  # ← 追加
from psycopg2.extras import execute_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import skeletonized satellite roads into PostGIS road_links"
    )
    parser.add_argument(
        "--skeleton-path",
        type=str,
        default="road_line_mask_skel_v3.png",
        help="Skeleton PNG path (1px width road lines)",
    )
    # 画像がカバーする緯度経度範囲（必須）
    parser.add_argument("--min-lat", type=float, required=True, help="南端（最小）緯度")
    parser.add_argument("--max-lat", type=float, required=True, help="北端（最大）緯度")
    parser.add_argument("--min-lon", type=float, required=True, help="西端（最小）経度")
    parser.add_argument("--max-lon", type=float, required=True, help="東端（最大）経度")

    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("NODO_DB_URL") or os.environ.get("DATABASE_URL"),
        help="Postgres connection URL (e.g. postgresql://user:pass@host:5432/dbname). "
             "If omitted, NODO_DB_URL or DATABASE_URL env is used.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="satellite",
        help="source カラムの値 (default: satellite)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="confidence カラムの値 (default: 0.3)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="ノイズ除去用: このポイント数未満のラインは捨てる",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DBに書き込まず、検出本数のみ表示",
    )
    return parser.parse_args()


# -----------------------------
# 画像座標 <-> 緯度経度 変換
# -----------------------------

def pixel_to_geocoord(
    x: int,
    y: int,
    width: int,
    height: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> Tuple[float, float]:
    """
    画像座標 (x, y) -> (lon, lat)

    画像:
      x: [0, width-1]  左->右
      y: [0, height-1] 上->下

    緯度:
      上端 = max_lat
      下端 = min_lat
    経度:
      左端 = min_lon
      右端 = max_lon
    """
    # 正規化座標
    nx = x / (width - 1)
    ny = y / (height - 1)

    lon = min_lon + (max_lon - min_lon) * nx
    # y が増えると南に向かうので max_lat から引く
    lat = max_lat - (max_lat - min_lat) * ny

    return lon, lat


# -----------------------------
# スケルトン画像 -> ポリライン抽出
# -----------------------------

def extract_polylines_from_skeleton(
    img: np.ndarray,
    min_points: int = 5,
) -> List[List[Tuple[int, int]]]:
    """
    スケルトン化された2値画像からポリライン（ピクセル座標列）を抽出する。

    アプローチ:
      - 非ゼロ画素を白(255)にしたバイナリ画像を作る
      - cv2.findContours を CHAIN_APPROX_NONE で実行
      - 各 Contour がポリラインになる（枝分かれ含めてOKと割り切る）

    戻り値:
      List[polyline], polyline は [(x, y), (x, y), ...]
    """
    # 非ゼロを255に
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    polylines: List[List[Tuple[int, int]]] = []

    for cnt in contours:
        if len(cnt) < min_points:
            continue
        poly: List[Tuple[int, int]] = []
        for pt in cnt:
            x, y = int(pt[0][0]), int(pt[0][1])
            poly.append((x, y))
        polylines.append(poly)

    return polylines


# -----------------------------
# PostGIS への書き込み
# -----------------------------

def build_linestring_wkt(coords: List[Tuple[float, float]]) -> str:
    """
    [(lon, lat), ...] -> 'LINESTRING(lon lat, ...)' のWKTを生成
    """
    coord_str = ", ".join(f"{lon} {lat}" for lon, lat in coords)
    return f"LINESTRING({coord_str})"


def insert_road_links(
    db_url: str,
    lines: List[List[Tuple[float, float]]],
    source: str,
    confidence: float,
    metadata: dict | None = None,
):
    if metadata is None:
        metadata = {}

    conn = psycopg2.connect(db_url)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            records = []
            for coords in lines:
                wkt = build_linestring_wkt(coords)
                metadata_json = json.dumps(metadata, ensure_ascii=False)
                # SQL のプレースホルダ順に合わせる: (wkt, metadata, source, confidence)
                records.append(
                    (
                        wkt,
                        metadata_json,
                        source,
                        confidence,
                    )
                )

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
            execute_batch(cur, sql, records, page_size=500)
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

    print(f"[INFO] Loading skeleton image from: {args.skeleton_path}")
    img = cv2.imread(args.skeleton_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Skeleton image not found: {args.skeleton_path}")

    height, width = img.shape
    print(f"[INFO] Image size: width={width}, height={height}")

    print("[INFO] Extracting polylines from skeleton...")
    polylines_px = extract_polylines_from_skeleton(
        img, min_points=args.min_points
    )
    print(f"[INFO] Extracted {len(polylines_px)} polylines (pixel space).")

    if not polylines_px:
        print("[WARN] No polylines found. Exit.")
        return

    print("[INFO] Converting pixel coordinates to geo coordinates...")
    lines_geo: List[List[Tuple[float, float]]] = []
    for poly in polylines_px:
        coords_geo: List[Tuple[float, float]] = []
        for x, y in poly:
            lon, lat = pixel_to_geocoord(
                x,
                y,
                width,
                height,
                min_lat=args.min_lat,
                max_lat=args.max_lat,
                min_lon=args.min_lon,
                max_lon=args.max_lon,
            )
            coords_geo.append((lon, lat))

        # 念のため、重複/ゼロ長を除去
        if len(coords_geo) >= 2:
            # 同一点連続はざっくり落とす
            deduped = [coords_geo[0]]
            for lon, lat in coords_geo[1:]:
                if abs(lon - deduped[-1][0]) < 1e-9 and abs(lat - deduped[-1][1]) < 1e-9:
                    continue
                deduped.append((lon, lat))

            if len(deduped) >= 2:
                lines_geo.append(deduped)

    print(f"[INFO] {len(lines_geo)} polylines left after cleaning (geo space).")

    if args.dry_run:
        print("[DRY RUN] Skipping DB insert.")
        return

    meta = {
        "source_image": os.path.basename(args.skeleton_path),
        "note": "imported from skeleton PNG",
    }

    print(f"[INFO] Inserting into PostGIS road_links (source={args.source}, confidence={args.confidence})...")
    insert_road_links(
        db_url=args.db_url,
        lines=lines_geo,
        source=args.source,
        confidence=args.confidence,
        metadata=meta,
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
