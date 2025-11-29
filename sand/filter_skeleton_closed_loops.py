# filter_skeleton_closed_loops.py

import cv2
import numpy as np

INPUT  = "road_line_mask_skel.png"          # 元の skeleton
OUTPUT = "road_line_mask_no_loops.png"      # 出力ファイル

# ★ 調整パラメータ（今回の指定値）
MAX_SMALL_LENGTH = 600      # これより短い線で
MAX_SMALL_ASPECT = 2.5      # アスペクト比がこれ未満なら「丸っこい」とみなす

def count_endpoints(component_mask: np.ndarray) -> int:
    """
    1コンポーネントの中で、neighbor が1個だけのピクセル（端点）の数を数える
    8近傍を使う
    """
    # padding して境界でも楽に近傍を見られるようにする
    padded = np.pad(component_mask, 1, mode="constant")
    ys, xs = np.nonzero(component_mask)

    endpoints = 0
    for y, x in zip(ys, xs):
        yy, xx = y + 1, x + 1  # padding 分ずらす
        window = padded[yy-1:yy+2, xx-1:xx+2]
        neighbors = int(window.sum() - padded[yy, xx])
        if neighbors == 1:
            endpoints += 1
    return endpoints


def main():
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    binary = (img > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    print("num_labels:", num_labels)

    out = np.zeros_like(binary)

    removed_loops = 0
    removed_round = 0
    kept = 0

    for i in range(1, num_labels):  # 0 は背景
        x, y, w, h, area = stats[i]

        component = (labels == i).astype(np.uint8)
        length = int(area)                 # 細線なので area ≒ 線の長さ(px)
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side

        endpoints = count_endpoints(component)

        # 1) 端点が0 → 完全に閉じたループ
        if endpoints == 0:
            removed_loops += 1
            continue

        # 2) 端点が少なく、丸っこくて短い線 → 小さな輪郭・森ノイズとみなす
        if endpoints <= 2 and aspect < MAX_SMALL_ASPECT and length < MAX_SMALL_LENGTH:
            removed_round += 1
            continue

        # それ以外は残す（道路候補）
        out[labels == i] = 1
        kept += 1

    print(f"kept components      : {kept}")
    print(f"removed closed loops : {removed_loops}")
    print(f"removed small-round  : {removed_round}")

    out_img = (out * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out_img)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
