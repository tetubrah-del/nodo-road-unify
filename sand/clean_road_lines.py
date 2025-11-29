# clean_road_lines.py
import cv2
import numpy as np
from skimage.morphology import skeletonize

INPUT = "road_line_mask.png"
OUTPUT_SKELETON = "road_line_mask_skel.png"
OUTPUT_CLEAN = "road_line_mask_clean.png"

# ここはあとで調整してOK
MIN_LENGTH = 80      # これより短い線分はノイズとして捨てる（px）
MIN_ASPECT = 2.0     # 細長さ（長辺/短辺）がこれ未満も捨てる

def main():
    # 1) 読み込み
    mask = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 2) 細線化（1px 幅の骨だけ残す）
    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary > 0)      # bool
    skel_u8 = (skel.astype(np.uint8)) * 255

    cv2.imwrite(OUTPUT_SKELETON, skel_u8)
    print("saved skeleton:", OUTPUT_SKELETON)

    # 3) ラベリング
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel_u8, connectivity=8
    )
    print("num_labels:", num_labels)

    clean = np.zeros_like(skel_u8)
    kept = 0

    for i in range(1, num_labels):  # 0 は背景
        x, y, w, h, area = stats[i]

        length = area               # 細線なので area ≒ 長さ(px)
        short_side = max(1, min(w, h))
        long_side = max(w, h)
        aspect_ratio = long_side / short_side

        # ---- ノイズ除去条件 ----
        if length < MIN_LENGTH:
            continue
        if aspect_ratio < MIN_ASPECT:
            continue

        clean[labels == i] = 255
        kept += 1

    print("kept components:", kept)
    cv2.imwrite(OUTPUT_CLEAN, clean)
    print("saved cleaned mask:", OUTPUT_CLEAN)


if __name__ == "__main__":
    main()
