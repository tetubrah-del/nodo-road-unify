#!/usr/bin/env python
import argparse
import json
import os

import cv2


def parse_args():
    p = argparse.ArgumentParser(
        description="1枚の航空画像を 2x2 に分割して保存し、各タイルの緯度経度BBoxを出力するスクリプト"
    )
    p.add_argument("--input", required=True, help="入力画像パス（例: data/aerial_tsujiwara_1km_z17.jpg）")
    p.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    p.add_argument("--min-lat", type=float, required=True)
    p.add_argument("--max-lat", type=float, required=True)
    p.add_argument("--min-lon", type=float, required=True)
    p.add_argument("--max-lon", type=float, required=True)
    p.add_argument(
        "--prefix",
        default=None,
        help="出力ファイル名のプレフィックス（省略時は入力ファイル名ベース）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {args.input}")

    h, w = img.shape[:2]
    print(f"[INFO] input image size: w={w}, h={h}")

    half_h = h // 2
    half_w = w // 2

    # 画像上での 2x2 タイル定義（y1:y2, x1:x2）
    tiles_pixel = {
        "NW": (0, half_h, 0, half_w),
        "NE": (0, half_h, half_w, w),
        "SW": (half_h, h, 0, half_w),
        "SE": (half_h, h, half_w, w),
    }

    # 緯度経度の中間点
    mid_lat = (args.min_lat + args.max_lat) / 2.0
    mid_lon = (args.min_lon + args.max_lon) / 2.0

    # 各タイルのBBox（ざっくり線形補間：このスケールなら十分）
    tiles_bbox = {
        "NW": {
            "min_lon": args.min_lon,
            "max_lon": mid_lon,
            "min_lat": mid_lat,
            "max_lat": args.max_lat,
        },
        "NE": {
            "min_lon": mid_lon,
            "max_lon": args.max_lon,
            "min_lat": mid_lat,
            "max_lat": args.max_lat,
        },
        "SW": {
            "min_lon": args.min_lon,
            "max_lon": mid_lon,
            "min_lat": args.min_lat,
            "max_lat": mid_lat,
        },
        "SE": {
            "min_lon": mid_lon,
            "max_lon": args.max_lon,
            "min_lat": args.min_lat,
            "max_lat": mid_lat,
        },
    }

    base = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    meta = {
        "input": args.input,
        "width": w,
        "height": h,
        "min_lat": args.min_lat,
        "max_lat": args.max_lat,
        "min_lon": args.min_lon,
        "max_lon": args.max_lon,
        "mid_lat": mid_lat,
        "mid_lon": mid_lon,
        "tiles": {},
    }

    for key, (y1, y2, x1, x2) in tiles_pixel.items():
        tile_img = img[y1:y2, x1:x2]
        out_path = os.path.join(args.output_dir, f"{base}_{key}.png")
        cv2.imwrite(out_path, tile_img)

        bbox = tiles_bbox[key]
        meta["tiles"][key] = {
            "path": out_path,
            "min_lat": bbox["min_lat"],
            "max_lat": bbox["max_lat"],
            "min_lon": bbox["min_lon"],
            "max_lon": bbox["max_lon"],
        }

        print(
            f"[INFO] tile {key}: {out_path} | "
            f"lon[{bbox['min_lon']}, {bbox['max_lon']}], "
            f"lat[{bbox['min_lat']}, {bbox['max_lat']}]"
        )

    # bbox情報をJSON保存
    meta_path = os.path.join(args.output_dir, f"{base}_tiles_bbox.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[INFO] bbox meta saved to: {meta_path}")


if __name__ == "__main__":
    main()
