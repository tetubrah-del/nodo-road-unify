#!/usr/bin/env python
"""
07_affine_mask_import.py

手描き道路マスク (0/255 PNG) から
- スケルトン抽出
- ポリライン化
- pixel -> lat/lon 変換（3点キャリブレーション済みアフィン変換）
- PostGIS (road_links) へ INSERT

※ 以前の 07_manual_mask_to_road_links.py を全面置き換え。
"""

import argparse
import json
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize
from shapely.geometry import LineString
from sqlalchemy import create_engine, text


# ==========================================================
# 0. 3点キャリブレーションからアフィン行列を構成
#    pixel(x,y) -> (lat, lon)
# ==========================================================

# 画像座標 (px)  ※ (x, y)
# 手描きマスク画像上の3点
_IMG_PTS = np.array([
    [4.0,   625.0],   # 左端
    [416.0, 334.0],   # 中央
    [762.0, 26.0],    # 右端
], dtype=np.float32)

# 対応する地理座標 (lat, lon)
_GEO_PTS = np.array([
    [33.1457325644, 131.5199521039],  # 左端
    [33.1483444856, 131.5245356460],  # 中央
    [33.1511452435, 131.5282460966],  # 右端
], dtype=np.float32)

# OpenCV の getAffineTransform は (x,y) -> (x',y') を想定
# 今回は (x,y) -> (lat, lon) として扱う
_AFFINE_M = cv2.getAffineTransform(_IMG_PTS, _GEO_PTS)  # shape (2,3)


def pixel_to_latlon_affine(x: float, y: float) -> tuple[float, float]:
    """
    pixel(x,y) -> (lat, lon) 変換。
    3点キャリブレーションで、回転＋スケール＋平行移動を吸収。
    """
    v = np.array([x, y, 1.0], dtype=np.float32)
    lat = float(_AFFINE_M[0].dot(v))
    lon = float(_AFFINE_M[1].dot(v))
    return lat, lon


# ==========================================================
# 1. 引数パーサ
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mask-path", required=True)
    p.add_argument("--db-url", required=True)
    p.add_argument("--width-m", type=float, default=3.0)
    p.add_argument("--min-points", type=int, default=5)

    # 微調整用のオフセット（基本は 0 のままでOK）
    p.add_argument("--delta-lat", type=float, default=0.0)
    p.add_argument("--delta-lon", type=float, default=0.0)

    return p.parse_args()


# ==========================================================
# 2. スケルトン -> ポリライン抽出（元のロジックを使用）
# ==========================================================
def skeleton_to_polylines(skel: np.ndarray):
    """
    skel: 0/1 or bool の細線化画像（8近傍）

    戻り値: list[list[(x, y)]]
    """
    h, w = skel.shape

    on_pixels = {(x, y) for y, x in zip(*np.where(skel > 0))}

    # 8近傍
    neighbors8 = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    def get_neighbors(x, y):
        res = []
        for dx, dy in neighbors8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) in on_pixels:
                res.append((nx, ny))
        return res

    # 各画素の次数
    degree = {p: len(get_neighbors(*p)) for p in on_pixels}

    visited = set()
    polylines = []

    # --- 端点（degree == 1）から伸ばす ---
    endpoints = [p for p, deg in degree.items() if deg == 1]

    def walk(start):
        path = [start]
        visited.add(start)
        prev = None
        cur = start

        while True:
            nbrs = [n for n in get_neighbors(*cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        return path

    for ep in endpoints:
        if ep in visited:
            continue
        path = walk(ep)
        if len(path) >= 2:
            polylines.append(path)

    # --- ループ部分（degree == 2 のみ）の処理 ---
    remaining = [p for p in on_pixels if p not in visited]
    for p in remaining:
        if p in visited:
            continue
        path = [p]
        visited.add(p)
        prev = None
        cur = p

        while True:
            nbrs = [n for n in get_neighbors(*cur) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == path[0]:
                path.append(nxt)
                break
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        if len(path) >= 2:
            polylines.append(path)

    return polylines


# ==========================================================
# 3. メイン処理
# ==========================================================
def main():
    args = parse_args()

    print(f"[INFO] Affine matrix (pixel -> lat,lon):")
    print(_AFFINE_M)

    # ---- マスク読み込み ----
    print(f"[INFO] Loading mask: {args.mask_path}")
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask_path)

    h, w = mask.shape[:2]
    print(f"[INFO] Mask size: {w} x {h}")

    # 0 以外を道路とみなす（手描きマスク）
    binary = (mask > 0).astype(np.uint8)

    # ---- スケルトン ----
    print("[INFO] Skeletonizing...")
    skel = skeletonize(binary > 0).astype(np.uint8)
    print(f"[INFO] Skeleton pixels: {int(np.count_nonzero(skel))}")

    # ---- ポリライン抽出 ----
    polylines_px = skeleton_to_polylines(skel)
    print(f"[INFO] Raw polylines: {len(polylines_px)}")

    # 最低点数でフィルタ
    polylines_px = [pl for pl in polylines_px if len(pl) >= args.min_points]
    print(f"[INFO] Filtered polylines (min_points={args.min_points}): {len(polylines_px)}")

    if not polylines_px:
        print("[WARN] No polylines found. Nothing to insert.")
        return

    # ---- delta オフセット情報 ----
    delta_lat = args.delta_lat
    delta_lon = args.delta_lon
    print(f"[INFO] delta_lat={delta_lat}, delta_lon={delta_lon}")

    # ---- pixel → (lon,lat) 変換 ----
    polylines_ll = []
    for pl in polylines_px:
        coords = []
        for (x, y) in pl:
            lat, lon = pixel_to_latlon_affine(float(x), float(y))
            lat += delta_lat
            lon += delta_lon
            coords.append((lon, lat))  # WKT は (lon lat)
        polylines_ll.append(coords)

    # ---- LineString に変換 ----
    lines_lonlat = []
    for coords in polylines_ll:
        if len(coords) < 2:
            continue
        lines_lonlat.append(LineString(coords))

    if not lines_lonlat:
        print("[WARN] No LineString objects created. Nothing to insert.")
        return

    # ---- DB 接続 ----
    print(f"[INFO] Connecting DB: {args.db_url}")
    engine = create_engine(args.db_url, future=True)

    # 既存の manual 行を削除してから再インポート
    delete_sql = text("DELETE FROM road_links WHERE source = 'manual';")

    insert_sql = text("""
        INSERT INTO road_links
        (geom,
         width_m,
         slope_deg,
         curvature,
         visibility,
         ground_condition,
         danger_score,
         metadata,
         source,
         confidence,
         parent_link_id)
        VALUES (
            ST_SetSRID(ST_GeomFromText(:wkt), 4326),
            :width_m,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            CAST(:metadata AS jsonb),
            'manual',
            :confidence,
            NULL
        )
        RETURNING link_id;
    """)

    # ---- DELETE + INSERT ループ ----
    with engine.begin() as conn:
        print("[INFO] Deleting existing manual links...")
        conn.execute(delete_sql)

        print("[INFO] Inserting new manual links...")
        for i, line in enumerate(lines_lonlat, start=1):
            meta = {
                "source_detail": "manual_mask_affine_v1",
                "mask_path": args.mask_path,
                "index": i,
            }
            params = {
                "wkt": line.wkt,
                "width_m": args.width_m,
                "metadata": json.dumps(meta),
                "confidence": 0.9,
            }
            res = conn.execute(insert_sql, params)
            link_id = res.scalar_one()
            print(f"[INFO] inserted link_id={link_id}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
