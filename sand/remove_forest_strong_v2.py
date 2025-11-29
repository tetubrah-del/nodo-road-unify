# remove_forest_strong_v2.py (no matplotlib)
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

INPUT = "data/aerial_no_veg.png"
MASK_OUT = "forest_mask_strong_v2.png"
OUTPUT = "aerial_no_forest_strong_v2.png"

# LBP パラメータ（大幅にゆるく）
RADIUS = 3       # 半径 1→3
NEIGHBORS = 16   # 近傍数 8→16
THRESH_FRAC = 0.60  # ヒストグラムの上位何％を森とみなすか


def main():

    img = cv2.imread(INPUT)
    if img is None:
        raise RuntimeError("input not found")

    # BGR → Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP 計算
    lbp = local_binary_pattern(gray, NEIGHBORS, RADIUS, method="uniform")

    # LBP ヒストグラムからしきい値自動推定
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    cumulative = np.cumsum(hist) / np.sum(hist)

    # 上位 THRESH_FRAC（60%）のしきい値を取得
    th_idx = np.where(cumulative > THRESH_FRAC)[0][0]
    threshold_value = th_idx

    print(f"Auto LBP threshold = {threshold_value}")

    # マスク生成：LBP値が大きいほどザワザワ＝森
    forest_mask = (lbp >= threshold_value).astype(np.uint8) * 255

    cv2.imwrite(MASK_OUT, forest_mask)
    print("Saved:", MASK_OUT)

    # マスクで森林を黒塗り
    masked = img.copy()
    masked[forest_mask == 255] = (0, 0, 0)

    cv2.imwrite(OUTPUT, masked)
    print("Saved:", OUTPUT)

    print("\n--- 完了 ---")


if __name__ == "__main__":
    main()
