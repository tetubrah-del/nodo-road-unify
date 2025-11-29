# filter_major_roads.py
import cv2
import numpy as np
from pathlib import Path

# 入出力ファイル
INPUT  = "road_line_mask_skel_v3_shape_filtered.png"
OUTPUT_MAJOR = "road_major_skel.png"
OUTPUT_COMBINED = "road_skel_with_major.png"

# パラメータ（ここをいじればチューニングできる）
MIN_LENGTH = 80      # 幹線候補とみなす最小長さ（ピクセル）
MIN_ASPECT = 4.0     # 細長さ（max(w,h)/max(1,min(w,h))) の下限
MAX_TORTUOSITY = 4.0 # くねくね度 = length / bbox_diag の上限

def main():
    path = Path(INPUT)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 0/1 に正規化
    binary = (img > 0).astype(np.uint8)

    # ラベリング
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    major_mask = np.zeros_like(binary)

    kept_major = 0
    kept_total = 0

    h, w = binary.shape

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area == 0:
            continue

        kept_total += 1

        # 成分ピクセル座標
        ys, xs = np.where(labels == i)
        length = len(xs)  # スケルトンなので area ≒ length

        # 長さフィルタ
        if length < MIN_LENGTH:
            continue

        # バウンディングボックス
        x0, y0, bw, bh, _ = stats[i]
        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))
        aspect = long_side / float(short_side)

        # bbox 対角線長
        bbox_diag = np.hypot(bw, bh)
        if bbox_diag < 1:
            continue
        tortuosity = length / bbox_diag  # 1 に近いほど直線

        # 条件判定
        if aspect >= MIN_ASPECT and tortuosity <= MAX_TORTUOSITY:
            major_mask[labels == i] = 1
            kept_major += 1

    print(f"total components: {kept_total}")
    print(f"kept major components: {kept_major}")

    # 出力（幹線のみ）
    major_out = (major_mask * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_MAJOR, major_out)
    print("saved:", OUTPUT_MAJOR)

    # もとのスケルトンに「幹線を少し太くして」重ねた可視化も保存
    major_dilated = cv2.dilate(major_out, np.ones((3, 3), np.uint8))
    combined = img.copy()
    combined[major_dilated > 0] = 255  # 幹線をより明るく
    cv2.imwrite(OUTPUT_COMBINED, combined)
    print("saved:", OUTPUT_COMBINED)


if __name__ == "__main__":
    main()
