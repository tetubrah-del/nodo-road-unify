# filter_skeleton_by_shape.py
#
# road_line_mask_skel_v3.png を読み込んで、
# 「線っぽさ」が低い成分（くるん・塊っぽい形）を消す。
#
# 出力: road_line_mask_skel_v3_shape_filtered.png

import cv2
import numpy as np

INPUT  = "road_line_mask_skel_v3.png"
OUTPUT = "road_line_mask_skel_v3_shape_filtered.png"

# ===== パラメータ（ここを調整） =====
MIN_LENGTH_KEEP      = 20    # これ未満はノイズとして捨てる
MIN_LENGTH_FOR_TEST  = 80    # これ以上の成分だけ thinness をチェック
THINNESS_MIN         = 0.08  # thinness がこれ未満なら「くるん寄り」とみなして捨てる
LONG_LINE_LEN        = 800   # これより長い成分は無条件で残す（幹線っぽいものを保護）
# ===============================

def main():
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    binary = (img > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print("components:", num_labels - 1)

    kept_mask = np.zeros_like(binary, dtype=np.uint8)
    kept_count = 0
    removed_loops = 0
    removed_small = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]   # = length (スケルトンなのでほぼ長さ)
        if area < MIN_LENGTH_KEEP:
            removed_small += 1
            continue

        # バウンディングボックス
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        bbox_area = max(1, w * h)
        thinness = float(area) / bbox_area  # 線ほど大きく、塊ほど小さい

        # 長い線は無条件で残す
        if area >= LONG_LINE_LEN:
            keep = True
        # そこそこ長い成分は thinness で判定
        elif area >= MIN_LENGTH_FOR_TEST:
            if thinness < THINNESS_MIN:
                keep = False
                removed_loops += 1
            else:
                keep = True
        else:
            # 短いけどノイズでないものは一旦残す
            keep = True

        if keep:
            kept_mask[labels == label] = 1
            kept_count += 1

    print(f"kept components:       {kept_count}")
    print(f"removed small (<{MIN_LENGTH_KEEP}): {removed_small}")
    print(f"removed loop-like:     {removed_loops}")

    out = (kept_mask * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
