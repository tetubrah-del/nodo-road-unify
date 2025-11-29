#!/usr/bin/env python
"""
Google 航空写真から道路スケルトンを抽出し、
PostGIS (road_links) に LineString としてインポートするスクリプト。

想定テーブル:
  road_links(
      link_id SERIAL PRIMARY KEY,
      geom        geometry(LineString, 4326),
      width_m     REAL,
      slope_deg   REAL,
      curvature   REAL,
      visibility  REAL,
      ground_condition INTEGER,
      danger_score REAL,
      source      TEXT,
      metadata    JSONB
  )

使い方の例:

  python 05_extract_roads_from_google.py \
    --image-path data/tsuji_google.png \
    --min-lat 33.14664436214888 \
    --max-lat 33.14870825466783 \
    --min-lon 131.51944979962673 \
    --max-lon 131.52423850947574 \
    --db-url "postgresql://nodo:nodo_password@localhost:5432/nodo" \
    --source-label "google_skeleton" \
    --min-pixels 50 \
    --dry-run
"""

import argparse
import json
from typing import List, Tuple, Dict, Set

import cv2
import numpy as np
import psycopg2
from skimage.morphology import skeletonize  # ★ scikit-image


# --------------------
#  引数
# --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True,
                   help="Google 航空写真の PNG/JPEG パス")
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument("--db-url", type=str, required=True,
                   help="Postgres 接続 URL (postgresql://...)")
    p.add_argument("--source-label", type=str, default="google_skeleton",
                   help="road_links.source に書き込む文字列")
    p.add_argument("--min-pixels", type=int, default=50,
                   help="このピクセル数未満のポリラインは捨てる(ノイズ除去)")
    p.add_argument("--dry-run", action="store_true",
                   help="DB に書き込まず、抽出本数だけをログ出力")
    return p.parse_args()


# --------------------
#  pixel <-> lat/lon
# --------------------

def pixel_to_latlon(
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
    画像ピクセル (x, y) -> (lon, lat)

    左上が (0,0), 右下が (width-1, height-1) 前提。
    lat は上が max_lat, 下が min_lat。
    """
    nx = x / (width - 1)
    ny = y / (height - 1)

    lon = min_lon + nx * (max_lon - min_lon)
    lat = max_lat - ny * (max_lat - min_lat)
    return lon, lat


# --------------------
#  2値化
# --------------------

def preprocess_for_roads(img_bgr: np.ndarray) -> np.ndarray:
    """
    Google 航空写真を道路抽出しやすいモノクロ画像に変換して 0/255 の2値画像を返す。
    道路は明るく、背景(田畑等)は暗くなるように補正。
    """
    # コントラスト少しアップ
    img = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 局所コントラスト
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ノイズ低減
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu しきい値で道路を白に
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 穴埋め＆平滑化
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    return th


# --------------------
#  スケルトン化 (scikit-image)
# --------------------

def skeletonize_fast(binary: np.ndarray) -> np.ndarray:
    """
    0/255 の2値画像 -> 0/255 スケルトン画像 (1px 幅)
    scikit-image の skeletonize を使用（高速）
    """
    # bool に変換
    binary_bool = binary > 0
    skel_bool = skeletonize(binary_bool)
    skel = (skel_bool.astype(np.uint8)) * 255
    return skel


# --------------------
#  スケルトン → ポリライン化
# --------------------

NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]


def skeleton_to_graph(skel: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    スケルトン画像(0/255)をグラフ構造に変換。
    ノード: ピクセル座標 (x, y)
    エッジ: 8近傍で繋がる前景ピクセル
    """
    h, w = skel.shape
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys, xs):
        node = (x, y)
        neighbors = []
        for dx, dy in NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] > 0:
                neighbors.append((nx, ny))
        if neighbors:
            graph[node] = neighbors
    return graph


def extract_polylines(graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
                      min_pixels: int) -> List[List[Tuple[int, int]]]:
    """
    グラフからポリライン(点列)のリストを抽出する。
    min_pixels 未満の短い線分は除外。
    """
    degree = {node: len(neigh) for node, neigh in graph.items()}

    visited: Set[Tuple[int, int]] = set()
    polylines: List[List[Tuple[int, int]]] = []

    endpoints = [n for n, d in degree.items() if d == 1]

    def walk(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [start]
        visited.add(start)
        current = start
        prev = None

        while True:
            neighbors = [n for n in graph.get(current, []) if n != prev]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            prev, current = current, nxt

        return path

    # 1) エンドポイントから
    for ep in endpoints:
        if ep in visited:
            continue
        poly = walk(ep)
        if len(poly) >= min_pixels:
            polylines.append(poly)

    # 2) 残り (ループ等)
    for node in graph.keys():
        if node in visited:
            continue
        poly = walk(node)
        if len(poly) >= min_pixels:
            polylines.append(poly)

    return polylines


# --------------------
#  DB 挿入
# --------------------

def insert_polylines_to_db(
    conn,
    polylines: List[List[Tuple[int, int]]],
    img_width: int,
    img_height: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    source_label: str,
):
    cur = conn.cursor()
    inserted = 0

    for poly in polylines:
        coords_lonlat: List[Tuple[float, float]] = []
        for x, y in poly:
            lon, lat = pixel_to_latlon(
                x, y,
                img_width, img_height,
                min_lat, max_lat,
                min_lon, max_lon
            )
            coords_lonlat.append((lon, lat))

        coord_str = ", ".join(f"{lon} {lat}" for lon, lat in coords_lonlat)
        wkt = f"LINESTRING({coord_str})"

        metadata = {
            "from": "google_image",
            "extract_script": "05_extract_roads_from_google.py",
        }
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        cur.execute(
            """
            INSERT INTO road_links (geom, source, metadata)
            VALUES (ST_GeomFromText(%s, 4326), %s, %s::jsonb)
            """,
            (wkt, source_label, metadata_json),
        )
        inserted += 1

    return inserted


# --------------------
#  main
# --------------------

def main():
    args = parse_args()

    # 画像ロード
    print(f"[INFO] Loading image: {args.image_path}")
    img = cv2.imread(args.image_path)
    if img is None:
        raise FileNotFoundError(args.image_path)
    h, w = img.shape[:2]
    print(f"[INFO] Image size: {w} x {h}")

    # ============================================================
    # 1) ここまでで SAM の道路マスク（0/255）を作っている前提
    #    変数名を sam_road_mask と仮定
    #    もし bool や 0/1 ならこの直前で 0/255 に変換しておく
    # ============================================================
    # sam_road_mask: shape = (h, w), dtype = uint8, 値は 0 or 255

    # ------------------------------------------------------------
    # 2) HSV で「農道寄りの道路色」を拾うマスクを作成（②）
    # ------------------------------------------------------------
    print("[INFO] Building HSV-based road mask ...")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 農道も拾えるように、彩度と明度の許容範囲を広めに取る
    # - s < 80 : グレー〜やや色付きまで
    # - 70 < v < 245 : 暗すぎる影と真っ白なハイライトを除外
    hsv_mask = np.zeros_like(s_ch, dtype=np.uint8)
    cond = (s_ch < 80) & (v_ch > 70) & (v_ch < 245)
    hsv_mask[cond] = 255

    print(f"[DEBUG] hsv_mask nonzero: {int(np.count_nonzero(hsv_mask))}")

    # ------------------------------------------------------------
    # 3) SAM マスクとの AND で「色も SAM も道路っぽい場所」に絞る
    # ------------------------------------------------------------
    if sam_road_mask.dtype != np.uint8:
        sam_road_mask_u8 = (sam_road_mask > 0).astype(np.uint8) * 255
    else:
        sam_road_mask_u8 = sam_road_mask

    combined_mask = cv2.bitwise_and(hsv_mask, sam_road_mask_u8)
    print(f"[DEBUG] combined_mask nonzero (before morph): {int(np.count_nonzero(combined_mask))}")

    # ------------------------------------------------------------
    # 4) 形態学的処理でノイズを削り、細長い構造を太らせる
    # ------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite("debug_combined_mask.png", combined_mask)
    print(f"[DEBUG] combined_mask nonzero (after morph): {int(np.count_nonzero(combined_mask))}")

    # ------------------------------------------------------------
    # 5) 連結成分ごとの「細長さフィルタ」（③）
    #    - 面積が小さいものを捨てる
    #    - アスペクト比（長辺/短辺）が低い＝丸い・四角いものを捨てる
    # ------------------------------------------------------------
    print("[INFO] Connected component filtering (area + aspect ratio)...")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined_mask, connectivity=8
    )

    filtered_mask = np.zeros_like(combined_mask)

    # パラメータはあとで調整用：
    MIN_AREA    = 80     # これより小さいブロブは捨てる
    MIN_ASPECT  = 2.5    # これより細長くないと「道路」とみなさない

    kept = 0
    for label in range(1, num_labels):  # 0 は背景
        x, y, w_cc, h_cc, area = stats[label]

        if area < MIN_AREA:
            continue

        # アスペクト比（長いほう / 短いほう）
        aspect = max(w_cc, h_cc) / max(1, min(w_cc, h_cc))

        if aspect < MIN_ASPECT:
            continue

        # 条件を満たしたコンポーネントだけ残す
        filtered_mask[labels == label] = 255
        kept += 1

    print(f"[INFO] CC kept: {kept} / {num_labels-1}")
    cv2.imwrite("debug_filtered_mask.png", filtered_mask)

    # ------------------------------------------------------------
    # 6) skeletonize 用の 0/1 バイナリへ変換
    # ------------------------------------------------------------
    binary = (filtered_mask > 0).astype(np.uint8)

    print("[INFO] Thinning with scikit-image.skeletonize ...")
    from skimage.morphology import skeletonize
    thin = skeletonize(binary).astype(np.uint8)   # 0 or 1

    skeleton = (thin * 255).astype(np.uint8)
    print(f"[DEBUG] skeleton nonzero: {int(np.count_nonzero(skeleton))}")
    cv2.imwrite("debug_skeleton.png", skeleton)

    # ------------------------------------------------------------
    # 7) ここから先は従来どおり：
    #    skeleton -> graph -> polylines -> DB へ保存
    # ------------------------------------------------------------
    print("[INFO] Building skeleton graph...")
    graph = skeleton_to_graph(skeleton)

    print(f"[INFO] Extracting polylines (min_pixels = {args.min_pixels})...")
    polylines_px = graph_to_polylines(graph, min_pixels=args.min_pixels)

    #   ・・・この先は元のコードのまま・・・

    print("[INFO] Extracted polylines:", len(polylines))

    if args.dry_run:
        print("[DRY RUN] 終了: DB には書き込んでいません。")
        return

    # 5) DB 書き込み
    print("[INFO] Connecting DB...")
    conn = psycopg2.connect(args.db_url)
    conn.autocommit = False

    try:
        # 既存を消したい場合はここをコメントアウト解除
        # with conn.cursor() as cur:
        #     cur.execute(
        #         "DELETE FROM road_links WHERE source = %s",
        #         (args.source_label,)
        #     )
        #     print(f"[INFO] Deleted {cur.rowcount} old rows for source={args.source_label}")

        print("[INFO] Inserting polylines into road_links ...")
        inserted = insert_polylines_to_db(
            conn,
            polylines,
            w, h,
            args.min_lat,
            args.max_lat,
            args.min_lon,
            args.max_lon,
            args.source_label,
        )
        conn.commit()
        print(f"[INFO] Inserted rows: {inserted}")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
