# filter_by_width_and_shape.py
import cv2
import numpy as np

INPUT = "road_surface_mask.png"
OUTPUT = "road_surface_width_filtered.png"

# ここが最重要！！（幅1.5〜3m ≒ 2px〜4px）
MIN_WIDTH_PX = 2
MAX_WIDTH_PX = 7  # 農道でも舗装路は6～8pxくらいに太ることが多い

def main():
    mask = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("failed to load mask")

    # ラベリング
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    out = np.zeros_like(mask)

    kept = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # component の短辺を「道路幅」とみなす
        short_side = min(w, h)
        long_side  = max(w, h)

        # 幅が農道レベルか？
        if MIN_WIDTH_PX <= short_side <= MAX_WIDTH_PX:
            # さらに細長さも軽く見る（田んぼの短辺＝大きすぎてフィルタ）
            aspect_ratio = long_side / (short_side + 1e-5)
            if aspect_ratio > 1.2:  # 軽めに設定（1.2以上なら線形）
                out[labels == i] = 255
                kept += 1

    print("kept components:", kept)
    cv2.imwrite(OUTPUT, out)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
