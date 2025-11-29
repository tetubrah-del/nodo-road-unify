# remove_forest_strong.py
#
# 入力:  data/aerial_no_veg.png
# 出力:  data/aerial_no_forest_strong.png   （森を黒塗りした画像）
#        debug_forest_mask_strong.png       （森と判定したマスク）

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ==== パラメータここだけいじればOK =======================
INPUT  = "data/aerial_no_veg.png"
OUTPUT = "data/aerial_no_forest_strong.png"
DEBUG_MASK = "debug_forest_mask_strong.png"

# LBP（テクスチャ）の設定
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS

# 森っぽい緑の色レンジ（HSV）
# 必要ならここを少しずつ広げて調整
GREEN_H_LO = 25
GREEN_H_HI = 95
GREEN_S_LO = 60
GREEN_V_LO = 40

# 森判定マスクの後処理
MORPH_KERNEL_SIZE = 5       # 形態素フィルタのカーネル
MIN_FOREST_REGION = 500     # 森として残す最小面積（小さいと田んぼが巻き込まれる）
# =======================================================


def main():
    img = cv2.imread(INPUT)
    if img is None:
        raise RuntimeError(f"failed to read: {INPUT}")

    h, w = img.shape[:2]
    print("input size:", w, "x", h)

    # --- 色ベースの「緑」マスク ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    green_mask = cv2.inRange(
        hsv,
        (GREEN_H_LO, GREEN_S_LO, GREEN_V_LO),
        (GREEN_H_HI, 255, 255)
    )
    # 0/255 → 0/1
    green_mask_bin = (green_mask > 0).astype(np.uint8)

    # --- LBP テクスチャマスク（ザラザラ＝森候補） ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lbp = local_binary_pattern(gray_blur, LBP_POINTS, LBP_RADIUS, method="uniform")
    # 0–255 に正規化
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu でテクスチャの強い領域（明るい側）を抽出
    _, lbp_mask = cv2.threshold(
        lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lbp_mask_bin = (lbp_mask > 0).astype(np.uint8)

    # --- 色 × テクスチャ の AND で「森確度の高い領域」 ---
    forest_mask = cv2.bitwise_and(green_mask_bin, lbp_mask_bin)

    # --- 形態素フィルタでマスクを整形 ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # --- 小さい領域を間引き（田んぼ・建物の巻き込み削減） ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        forest_mask, connectivity=8
    )

    cleaned = np.zeros_like(forest_mask)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_FOREST_REGION:
            continue
        cleaned[labels == i] = 1
        kept += 1

    print(f"forest components kept: {kept}")

    forest_mask_u8 = (cleaned * 255).astype(np.uint8)
    cv2.imwrite(DEBUG_MASK, forest_mask_u8)
    print("saved forest mask:", DEBUG_MASK)

    # --- 森と判定したピクセルを黒で塗りつぶし ---
    out = img.copy()
    out[cleaned > 0] = (0, 0, 0)

    cv2.imwrite(OUTPUT, out)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
