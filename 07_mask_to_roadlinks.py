#!/usr/bin/env python
"""
クリック式 SAM で作った道路マスク（PNG）を
skeletonize → LineString → PostGIS road_links に INSERT する。

例:

python 07_mask_to_roadlinks.py \
  --image-path data/tsuji_google.png \
  --mask-path data/tsuji_roads_mask_click.png \
  --min-lat 33.14763163983619 \
  --max-lat 33.15231274805454 \
  --min-lon 131.51580079556032 \
  --max-lon 131.52656086836566 \
  --db-url "postgresql://nodo:nodo_password@localhost:5432/nodo" \
  --source sam_click_google
"""

import argparse
import json

import cv2
import numpy as np
from skimage.morphology import skeletonize
from shapely.geometry import LineString
import psycopg2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True)
    p.add_argument("--mask-path", required=True)
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--db-url", required=True)
    p.add_argument("--source", default="sam_click_google")
    p.add_argument("--min-pixels", type=int, default=15,
                   help="最低ピクセル数（短すぎる線を除外）")
    p.add_argument("--simplify-tol", type=float, default=0.0,
                   help="LineString.simplify のトレランス（0 なら無効）")
    return p.parse_args()


def pixel_to_lonlat(x, y, width, height, min_lon, max_lon, min_lat, max_lat):
    lon = min_lon + (max_lon - min_lon) * (x / (width - 1))
    lat = max_lat - (max_lat - min_lat) * (y / (height - 1))
    return lon, lat


def skeleton_to_lines(mask_bin: np.ndarray,
                      min_pixels: int,
                      width: int,
                      height: int,
                      min_lon: float,
                      max_lon: float,
                      min_lat: float,
                      max_lat: float):
    """2値マスク → skeleton → LineString のリスト"""
    # skeletonize は bool 前提
    skel = skeletonize(mask_bin > 0)
    skel_u8 = (skel.astype(np.uint8) * 255)

    # ラベリング
    num_labels, labels = cv2.connectedComponents(skel_u8)
    print(f"[INFO] Skeleton components: {num_labels - 1}")

    lines = []

    for lab in range(1, num_labels):
        comp = (labels == lab).astype(np.uint8) * 255
        # 各コンポーネントに対して contour を取得
        contours, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for cnt in contours:
            pts = cnt[:, 0, :]  # (N, 2)
            if len(pts) < min_pixels:
                continue

            # ピクセル → lon/lat
            coords = [
                pixel_to_lonlat(
                    int(x), int(y),
                    width, height,
                    min_lon, max_lon,
                    min_lat, max_lat
                )
                for (x, y) in pts
            ]
            if len(coords) < 2:
                continue

            line = LineString(coords)
            lines.append(line)

    print(f"[INFO] Extracted LineStrings: {len(lines)}")
    return lines


def insert_lines_to_postgis(lines,
                            db_url: str,
                            source: str):
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # source カラムが無ければ Warn だけ出す（ALTER はユーザー側で一度やっておけばOK）
    try:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='road_links' AND column_name='source';")
        has_source = cur.fetchone() is not None
    except Exception as e:
        print(f"[WARN] failed to check 'source' column: {e}")
        has_source = False

    inserted = 0

    for line in lines:
        wkt = line.wkt
        if has_source:
            cur.execute(
                """
                INSERT INTO road_links (geom, source, metadata)
                VALUES (ST_SetSRID(ST_GeomFromText(%s), 4326), %s, %s)
                """,
                (wkt, source, json.dumps({"from": "sam_click"})),
            )
        else:
            cur.execute(
                """
                INSERT INTO road_links (geom, metadata)
                VALUES (ST_SetSRID(ST_GeomFromText(%s), 4326), %s)
                """,
                (wkt, json.dumps({"from": "sam_click", "source": source})),
            )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"[INFO] Inserted {inserted} road_links.")


def main():
    args = parse_args()

    # 画像サイズ取得（マスクと画像サイズが一致してる前提）
    img = cv2.imread(args.image_path)
    if img is None:
        raise FileNotFoundError(args.image_path)
    height, width = img.shape[:2]

    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask_path)

    if mask.shape[0] != height or mask.shape[1] != width:
        raise ValueError(
            f"Mask size {mask.shape[1]}x{mask.shape[0]} != "
            f"image size {width}x{height}"
        )

    mask_bin = (mask > 0).astype(np.uint8)
    print(f"[INFO] Nonzero mask pixels: {int(mask_bin.sum())}")

    lines = skeleton_to_lines(
        mask_bin=mask_bin,
        min_pixels=args.min_pixels,
        width=width,
        height=height,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
    )

    if not lines:
        print("[WARN] No lines extracted. Abort.")
        return

    insert_lines_to_postgis(
        lines=lines,
        db_url=args.db-url if False else args.db_url,  # safe: just args.db_url
        source=args.source,
    )


if __name__ == "__main__":
    main()
