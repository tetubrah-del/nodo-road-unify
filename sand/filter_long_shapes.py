# filter_long_shapes.py
import cv2
import numpy as np

INPUT = "road_surface_mask.png"
OUTPUT = "road_surface_long_only.png"

def main():
    mask = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read {INPUT}")

    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ラベリング
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    out = np.zeros_like(bin_mask)

    kept = 0
    MIN_AREA = 50   # ゴミ除去用の最小面積（そのまま）

    print("components:", num_labels)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA:
            continue

        # 長辺 / 短辺
        aspect_ratio = max(w, h) / max(1, min(w, h))
        # 占有率（1に近い → 四角い塊、0に近い → 細線）
        extent = area / float(w * h)

        # ---- フィルタ条件（ゆるめ）----
        # ・そこそこ細長い
        # ・あまり中身が詰まりすぎていない
        if aspect_ratio > 1.5 and extent < 0.6:
            out[labels == i] = 255
            kept += 1

    print("kept components:", kept)
    cv2.imwrite(OUTPUT, out)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
