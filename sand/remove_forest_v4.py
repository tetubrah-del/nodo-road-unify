# remove_forest_v4.py
import cv2
import numpy as np
from pathlib import Path

INPUT  = Path("data/aerial_tsujiwara_1km_z17.jpg")
MASK_OUT = Path("forest_mask_v4.png")
OUTPUT = Path("aerial_no_forest_v4.png")

# ==== 調整用パラメータ ====
# 「どれくらい緑なら森とみなすか」
GREEN_DIFF_THR = 18   # G - max(R,B) がこの値より大きい
GREEN_MIN      = 60   # G の絶対値がこれ以上
BRIGHT_MAX     = 170  # 明るさ上限（暗めの緑だけ森扱い）
SAT_MIN        = 40   # 彩度の下限（くすんだ灰色は除外）

# マスクを少しなめらかにする
MORPH_KERNEL = 5      # 形態素フィルタのカーネルサイズ
MORPH_ITERS  = 1      # くっつけ具合

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    img = cv2.imread(str(INPUT))
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    h, w = img.shape[:2]
    print("input size:", w, "x", h)

    # --- RGB/BGR から「緑成分」を見る ---
    b, g, r = cv2.split(img)

    # 緑の強さ（緑が他チャンネルよりどれだけ強いか）
    max_rb = np.maximum(r, b)
    greenness = g.astype(np.int16) - max_rb.astype(np.int16)

    # 明るさ & 彩度も見るために HSV も併用
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 条件を組み合わせて「森っぽい」ピクセルを判定
    forest_mask = (
        (greenness > GREEN_DIFF_THR) &  # 緑が他より十分強い
        (g > GREEN_MIN) &              # 緑チャンネルもある程度明るい
        (v_ch < BRIGHT_MAX) &          # 暗めの緑（＝森っぽい）
        (s_ch > SAT_MIN)               # ある程度彩度あり
    )

    forest_mask = forest_mask.astype(np.uint8) * 255

    # マスクを少し膨張・収縮して穴やギザギザをならす
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    forest_mask = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERS)

    # デバッグ用にマスク画像を保存（白=森）
    cv2.imwrite(str(MASK_OUT), forest_mask)
    print("saved forest mask:", MASK_OUT)

    # 森を黒塗り（または完全に消す）
    no_forest = img.copy()
    no_forest[forest_mask > 0] = (0, 0, 0)  # 森を黒に

    cv2.imwrite(str(OUTPUT), no_forest)
    print("saved no-forest image:", OUTPUT)


if __name__ == "__main__":
    main()
