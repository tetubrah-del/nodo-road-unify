#!/usr/bin/env python
"""
Planet のスクリーンショットから
「太い道路だけ」をざっくり抽出して可視化するテストスクリプト。

- 入力: RGB 衛星画像 (例: data/tsujiharu_Planet_screenshot.png)
- 出力:
    out_prefix_mask.png      : 2値マスク（太い道路っぽい領域）
    out_prefix_overlay.png   : 元画像に赤で太い道路を重ねた画像
"""

import argparse
import os

import cv2
import numpy as np


def extract_fat_roads(img_bgr: np.ndarray,
                      blur_size: int = 5,
                      open_kernel: int = 7,
                      min_area: int = 200) -> np.ndarray:
    """
    太い道路っぽい領域だけを 255 (白) で残したマスクを返す。

    - blur_size    : ノイズ除去用のガウシアンカーネルサイズ（奇数）
    - open_kernel  : 「細い線を消す」ためのオープニングカーネル（奇数, 7〜11 推奨）
    - min_area     : 連結成分の最小ピクセル数（小さいほど細い線も残る）
    """
    # 1) コントラスト強調（Planet スクショが白っぽいので少し締める）
    alpha = 1.3
    beta = 10
    img_adj = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    # 2) グレースケール + 軽いブラー
    gray = cv2.cvtColor(img_adj, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # 3) 大まかに「明るい領域」を抽出（道路＋裸地の候補）
    #    道路が周囲より明るい前提で Otsu で二値化
    _, binary = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4) 細い線（畦・区画線）を消すため、やや大きめカーネルでオープニング
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) ギャップを埋めるために軽くクロージング
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) 連結成分で「面積の小さいチリ」を捨てる
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    mask = np.zeros_like(closed)
    for label in range(1, num_labels):  # 0 は背景
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == label] = 255

    return mask


def make_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    元画像の上に、mask==255 の部分を赤で重ねた画像を返す。
    """
    overlay = img_bgr.copy()
    overlay[mask == 255] = (0, 0, 255)  # 赤 (BGR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Planet スクショ画像のパス (例: data/tsujiharu_Planet_screenshot.png)",
    )
    parser.add_argument(
        "--out-prefix",
        default="out/tsujiharu_roads",
        help="出力ファイル名のプレフィックス",
    )
    parser.add_argument("--blur-size", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)

    print("[INFO] Image size:", img.shape[1], "x", img.shape[0])

    mask = extract_fat_roads(
        img,
        blur_size=args.blur_size,
        open_kernel=args.open_kernel,
        min_area=args.min_area,
    )
    overlay = make_overlay(img, mask)

    mask_path = f"{args.out_prefix}_mask.png"
    overlay_path = f"{args.out_prefix}_overlay.png"

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

    n_pixels = int(mask.sum() / 255)
    print(f"[INFO] Saved mask to {mask_path}")
    print(f"[INFO] Saved overlay to {overlay_path}")
    print(f"[INFO] White pixels in mask: {n_pixels}")


if __name__ == "__main__":
    main()
