# make_road_surface_mask.py
import cv2
import numpy as np
from pathlib import Path

# 入力画像（1km, zoom17 の航空写真）
INPUT = "data/aerial_tsujiwara_1km_z17.jpg"
OUTPUT = "road_surface_mask.png"

def main():
    path = Path(INPUT)
    assert path.exists(), f"input image not found: {path}"

    bgr = cv2.imread(str(path))
    h, w, _ = bgr.shape
    print("input:", bgr.shape)

    # 1) HSV で「グレーっぽい明るい部分」を抽出
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # ---- ここがチューニングポイント ----
    # S: 0〜255 のうち 60 以下 → 低彩度（グレー, コンクリ）
    # V: 80〜230 → 暗すぎず、明るすぎない
    road_like = (S < 60) & (V > 80) & (V < 230)

    color_mask = np.zeros((h, w), np.uint8)
    color_mask[road_like] = 255

    # 2) グレースケール＋Canny でエッジ抽出
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)

    # 3) 「色が道路っぽい」かつ「エッジでもある」ところを残す
    combined = cv2.bitwise_and(edges, color_mask)

    # 4) 少し太らせて“面”にする
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(combined, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 小さすぎるゴミを削る（面積フィルタ）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    MIN_AREA = 30  # px 単位。足りなければ 10〜20 に下げてもOK

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_AREA:
            cleaned[labels == i] = 255

    cv2.imwrite(OUTPUT, cleaned)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
