#!/usr/bin/env python
"""
衛星由来の LineString に対して、
GPT Vision で「農道かどうか」「属性」を推定して metadata に保存する。

使い方例:

  export NODO_DB_URL="postgresql://nodo:nodo_password@localhost:5432/nodo"
  export OPENAI_API_KEY="..."

  python 08_gpt_filter_satellite_segments.py \
    --aerial-path data/tsujiharu_Planet_screenshot.png \
    --min-lat 33.14599 \
    --max-lat 33.14914 \
    --min-lon 131.52008 \
    --max-lon 131.52653 \
    --limit 50
"""

import os
import json
import base64
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from openai import OpenAI


# ==============================
#  座標変換 (lat/lon ↔ pixel)
# ==============================

import math

def lat_to_merc(lat):
    lat_rad = math.radians(lat)
    return math.log(math.tan(lat_rad/2 + math.pi/4))

def lon_to_merc(lon):
    return math.radians(lon)  # 緯度ほど歪んでいない（ほぼ線形）

def latlon_to_pixel(lon, lat, width, height,
                    min_lat, max_lat, min_lon, max_lon):

    # 1. すべて Mercator 空間にマッピング
    y_min = lat_to_merc(min_lat)
    y_max = lat_to_merc(max_lat)
    y    = lat_to_merc(lat)

    x_min = lon_to_merc(min_lon)
    x_max = lon_to_merc(max_lon)
    x     = lon_to_merc(lon)

    # 2. Mercator 空間で正確に線形補間
    nx = (x - x_min) / (x_max - x_min)
    ny = (y_max - y) / (y_max - y_min)

    px = int(nx * (width - 1))
    py = int(ny * (height - 1))
    return px, py


# ==============================
#  DB から衛星ラインを取得
# ==============================

def fetch_satellite_links(
    conn,
    limit: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
):
    """
    ★テスト用版：
    source 条件を外して、とにかく bbox 内の LineString を拾う。
    （本番では satellite だけに戻す）
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              link_id,
              ST_AsText(geom) AS wkt
            FROM road_links
            WHERE geom IS NOT NULL
              AND ST_Intersects(
                    geom,
                    ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                  )
            ORDER BY link_id
            LIMIT %s
            """,
            (min_lon, min_lat, max_lon, max_lat, limit),
        )
        return cur.fetchall()



# ==============================
#  WKT -> (lon,lat) リスト
# ==============================

def parse_linestring_wkt(wkt: str) -> List[Tuple[float, float]]:
    """
    'LINESTRING(lon lat, lon lat, ...)' をパースして [(lon, lat), ...] にする簡易パーサ
    """
    assert wkt.startswith("LINESTRING(") and wkt.endswith(")")
    body = wkt[len("LINESTRING("): -1]
    pts = []
    for part in body.split(","):
        lon_str, lat_str = part.strip().split()
        pts.append((float(lon_str), float(lat_str)))
    return pts


# ==============================
#  画像タイル切り出し＆線を描画
# ==============================

def make_segment_tile(
    base_img: np.ndarray,
    coords_lonlat: List[Tuple[float, float]],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    tile_size: int = 1024,
    pad: int = 192,
) -> np.ndarray:
    """
    航空写真から、そのライン周辺だけを tile_size x tile_size で切り出し、
    その上に赤線で LineString を描画した画像を返す。

    改訂点:
      - tile_size を 512 -> 1024 に拡大
      - pad を 32 -> 192 に拡大し、周囲コンテキストを多めに含める
    """
    h, w, _ = base_img.shape

    # ライン上の全点を pixel に変換
    pix_pts = [
        latlon_to_pixel(lon, lat, w, h, min_lat, max_lat, min_lon, max_lon)
        for lon, lat in coords_lonlat
    ]

    xs = [p[0] for p in pix_pts]
    ys = [p[1] for p in pix_pts]

    # バウンディングボックス＋余白
    x_min = max(min(xs) - pad, 0)
    x_max = min(max(xs) + pad, w - 1)
    y_min = max(min(ys) - pad, 0)
    y_max = min(max(ys) + pad, h - 1)

    crop = base_img[y_min:y_max + 1, x_min:x_max + 1].copy()

    # 座標を crop 内のローカル座標に変換
    local_pts = [(x - x_min, y - y_min) for (x, y) in pix_pts]

    # サイズを tile_size にリサイズ
    crop_h, crop_w, _ = crop.shape
    crop_resized = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

    # 座標もスケールを合わせる
    scale_x = tile_size / crop_w
    scale_y = tile_size / crop_h
    local_pts_scaled = [
        (int(px * scale_x), int(py * scale_y)) for (px, py) in local_pts
    ]

    # 赤線で描画
    for i in range(len(local_pts_scaled) - 1):
        cv2.line(
            crop_resized,
            local_pts_scaled[i],
            local_pts_scaled[i + 1],
            (0, 0, 255),  # BGR: 赤
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    return crop_resized


# ==============================
#  GPT へ問い合わせ
# ==============================

def classify_with_gpt(client: OpenAI, model: str, image_bgr: np.ndarray) -> dict:
    """
    OpenAI GPT（Vision対応）で画像を分類。
    赤線でハイライトされた部分が何か＋属性を JSON で返してもらう。
    """

    # BGR -> RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # PNG バイナリにエンコード
    ok, png_bytes = cv2.imencode(".png", img_rgb)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")

    BASE_PROMPT = """
あなたは衛星画像から「農道（FARMLANE）」だけを抽出する専門家です。
与えられた画像は、ある小さな範囲を切り出した衛星写真で、
その中心付近に 1 本の候補ラインが通っています（このラインをどのクラスにするか判定します）。

必ず、下記いずれか 1 つの class を選んでください：
- FARMLANE       : 農機・軽トラなど車両が実際に通行する農道
- FIELD_BOUNDARY : 田んぼ・畑の区画線や畦・畝。人は歩けるかもしれないが車両は通らない境界線
- RIVER          : 川・用水路（幅のある水の流れ）
- FOREST_EDGE    : 森林と畑・草地の境目の「木の帯」のような線
- OTHER          : 上記どれにもはっきり当てはまらないもの（生活道路・県道・建物の間の路地・建物の輪郭など）

### 判定ルール（とにかく FARMLANE を絞り込むこと）

1. FARMLANE（農道）と判断してよい条件
  - 農地（田んぼ・畑）の中や、農地と農地の間を通っている。
  - 幅がそこそこあり、農業機械や軽トラが通れる太さに見える。
  - ある程度の長さがあり、1枚の画像の中で「短い点線」ではなく、連続した線になっている。
  - 住宅街のメイン道路ではなく、「農地にアクセスするための道」に見える。
  - 道の両側に農地がある、あるいは農地に向かって伸びている。

2. FIELD_BOUNDARY（畦・畝・区画線）と判断すべき典型パターン
  - 細くて短い線が格子状・碁盤の目状にたくさん並んでいる。
  - 車が通るには明らかに細すぎる。
  - 田んぼや畑を小さく区切っているように見える。
  - 画像全体に「細い白い線がびっしり」あるだけで、どれも車道には見えない。

3. OTHER とすべき典型パターン
  - 住宅街・集落の中の舗装道路（生活道路・市道・県道など）。
  - 建物の輪郭、工場敷地の境界線、駐車場の線など。
  - 農地とあまり関係がなさそうな線。
  - 車は通れるが、明らかに「農道」ではなく一般道路に見えるもの。

4. RIVER / FOREST_EDGE の判断
  - RIVER：周囲と色が違う帯状のエリアで、湾曲しながら流れている。橋がかかっていることもある。
  - FOREST_EDGE：片側が濃い緑（森林）、片側が畑や草地で、その境目に沿って伸びる帯状のライン。

5. 「長さ・スケール」の目安
  - 画像の中で極端に短い線（ごく一部分しか写っていない）は、基本的に FARMLANE にしない。
  - 田んぼの区画の一辺だけが見えているような線は FIELD_BOUNDARY または OTHER とする。
  - 「これは農道かもしれないが、かなり短くて確信が持てない」場合も、FARMLANE ではなく OTHER か FIELD_BOUNDARY にすること。

6. 不確実なときの方針
  - 迷ったら FARMLANE にしない。
  - 自信がない場合は class を OTHER にし、confidence を低め（0.5〜0.6 程度）にする。

### width_category
- "NARROW" : 人かバイク程度が通れそうな細い道。軽トラ・農機にはギリギリ／厳しそう。
- "MEDIUM" : 軽トラ・小型車・小型農機が普通に通れそうな幅。
- "WIDE"   : 2車線道路や大型車も余裕で通れそうな幅。明らかに一般道路・幹線道路レベル。

### surface_type
- "PAVED"   : アスファルトやコンクリートで舗装されている。
- "GRAVEL"  : 砂利・土っぽい。
- "DIRT"    : 完全な土の道、あるいは草地の上にタイヤ跡がある程度。
- "UNKNOWN" : 判定困難な場合。

### visibility / tree_cover
- visibility: 0.0〜1.0 で、その道がどれくらいはっきり見えているか。
- tree_cover: 0.0〜1.0 で、その道の直近にどれくらい樹木が被さっているか。

### 出力フォーマット

出力は必ず次の JSON 1オブジェクトのみとする：

{
  "class": "FARMLANE or FIELD_BOUNDARY or RIVER or FOREST_EDGE or OTHER",
  "confidence": 0.0〜1.0,
  "width_category": "NARROW or MEDIUM or WIDE or UNKNOWN",
  "surface_type": "PAVED or GRAVEL or DIRT or UNKNOWN",
  "visibility": 0.0〜1.0,
  "tree_cover": 0.0〜1.0,
  "reason": "日本語で1〜2文の説明"
}
"""

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a vision model that returns strict JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        },
                    },
                ],
            },
        ],
    )

    content = resp.choices[0].message.content
    return json.loads(content)


# ==============================
#  metadata を UPDATE
# ==============================

def update_link_metadata(conn, link_id: int, gpt_result: dict, model_name: str):
    with conn.cursor() as cur:
        json_str = json.dumps(
            {
                "gpt_filter": {
                    **gpt_result,
                    "model": model_name,
                }
            },
            ensure_ascii=False,
        )

        cur.execute(
            """
            UPDATE road_links
            SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
            WHERE link_id = %s
            """,
            (json_str, link_id),
        )


# ==============================
#  main
# ==============================

def main():
    # ---- 引数パース ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--aerial-path", required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--limit", type=int, default=999999)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--db-url",
        default=os.getenv("NODO_DB_URL") or os.getenv("DATABASE_URL"),
        help="PostGIS の接続URL"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="GPT vision model name (e.g. gpt-4.1-mini)"
    )

    args = parser.parse_args()

    if not args.db_url:
        raise RuntimeError("DB URL がありません。")

    # ---- OpenAI クライアント ----
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")
    client = OpenAI(api_key=api_key)

    # ---- 画像ロード ----
    print("[INFO] Loading aerial image:", args.aerial_path)
    base_img = cv2.imread(args.aerial_path, cv2.IMREAD_COLOR)
    if base_img is None:
        raise FileNotFoundError(args.aerial_path)

    # Planetスクショのコントラスト調整
    alpha = 1.3
    beta = 10
    base_img = cv2.convertScaleAbs(base_img, alpha=alpha, beta=beta)

    h, w, _ = base_img.shape
    print(f"[INFO] Aerial image size: {w} x {h}")

    # ---- DB 接続 ----
    conn = psycopg2.connect(args.db_url)
    conn.autocommit = False

    # dry-run 出力フォルダ
    debug_dir = "debug_tiles"
    if args.dry_run:
        os.makedirs(debug_dir, exist_ok=True)

    try:
        # ------------------------
        #    対象ライン取得
        # ------------------------
        links = fetch_satellite_links(
            conn,
            args.limit,
            args.min_lat,
            args.max_lat,
            args.min_lon,
            args.max_lon,
        )
        print(f"[INFO] Fetched {len(links)} satellite links to process.")

        # ------------------------
        #    メインループ
        # ------------------------
        for idx, row in enumerate(links):
            link_id = row["link_id"]
            coords = parse_linestring_wkt(row["wkt"])

            tile = make_segment_tile(
                base_img,
                coords,
                min_lat=args.min_lat,
                max_lat=args.max_lat,
                min_lon=args.min_lon,
                max_lon=args.max_lon,
                tile_size=1024,
                pad=192,
            )

            # ========= dry-run ==========
            if args.dry_run:
                if idx < 30:  # とりあえず最初の30枚だけ保存
                    out_path = f"{debug_dir}/tile_{idx:02d}_link_{link_id}.png"
                    cv2.imwrite(out_path, tile)
                    print(f"[DRY RUN] Saved tile for link_id={link_id} -> {out_path}")
                continue
            # ============================

            # GPT 呼び出し
            print(f"[INFO] Calling GPT for link_id={link_id} ...")
            result = classify_with_gpt(client, args.model, tile)
            print(f"[INFO] GPT result for {link_id}: {result}")

            update_link_metadata(conn, link_id, result, args.model)
            conn.commit()

        print("[INFO] Done.")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
