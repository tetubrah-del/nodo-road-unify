# filter_skeleton_by_density.py
import cv2
import numpy as np

INPUT  = "road_line_mask_skel.png"
OUTPUT = "road_line_mask_skel_filtered.png"

# パラメータ
# - KERNEL_SIZE : 近傍サイズ（奇数）。大きいほど「広い範囲の密集」を検出
# - DENSITY_MAX : 近傍内の白ピクセル数がこれより多いと「森扱い」で削除
# - MIN_COMPONENT : 最後に残す連結成分の最小ピクセル数（小さいと細かいノイズが残る）
KERNEL_SIZE = 17
DENSITY_MAX = 40
MIN_COMPONENT = 40

def main():
    skel = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 0/1 のバイナリ
    binary = (skel > 0).astype(np.uint8)

    # 近傍の白ピクセル数をカウント（ボックスフィルタ）
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.float32)
    density = cv2.filter2D(
        binary.astype(np.float32),
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )

    # 密度が高すぎる部分（森っぽいザワザワ）をマスク
    sparse_mask = (density <= DENSITY_MAX).astype(np.uint8)

    # skeleton と AND を取ることで、密なところだけ削除
    filtered = cv2.bitwise_and(binary, binary, mask=sparse_mask)

    # 小さすぎる成分を削除（孤立ノイズ対策）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        filtered,
        connectivity=8
    )
    cleaned = np.zeros_like(filtered)

    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT:
            continue
        cleaned[labels == i] = 1
        kept += 1

    print(f"kept components: {kept}")

    out = (cleaned * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
