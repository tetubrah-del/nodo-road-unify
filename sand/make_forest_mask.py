# make_forest_mask.py
import cv2
import numpy as np

INPUT = "data/aerial_tsujiwara_1km_z17.jpg"
OUTPUT_FOREST = "forest_mask.png"
OUTPUT_CLEAR = "aerial_without_forest.png"

def main():
    img = cv2.imread(INPUT)
    if img is None:
        raise RuntimeError("failed to read input image")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === 緑色の森を抽出する（かなり強めの条件） ===
    # hue(色相) 35〜85 付近が “植物”
    lower = np.array([35, 40, 20])
    upper = np.array([85, 255, 255])

    forest_mask = cv2.inRange(hsv, lower, upper)

    # 保存（デバッグ用）
    cv2.imwrite(OUTPUT_FOREST, forest_mask)

    # === 森を黒塗りにした画像を作る ===
    clear = img.copy()
    clear[forest_mask > 0] = (0, 0, 0)

    cv2.imwrite(OUTPUT_CLEAR, clear)

    print("saved:", OUTPUT_FOREST)
    print("saved:", OUTPUT_CLEAR)

if __name__ == "__main__":
    main()
