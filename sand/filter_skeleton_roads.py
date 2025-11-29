import cv2
import numpy as np
from skimage import measure
from shapely.geometry import LineString

INPUT = "road_mask_tsujiwara_skel.png"
OUTPUT = "road_network_filtered.png"

# パラメータ（あとで調整する前提）
MIN_POINTS = 80        # ← 150 → 80 に緩和
MIN_LENGTH = 30.0      # ← 60 → 30 に緩和
MIN_ELONGATION = 2.0   # ← 4.0 → 2.0 に緩和


def compute_elongation(coords_xy: np.ndarray) -> float:
    """
    coords_xy: (N, 2) で [x, y]
    PCA の固有値比(λ1/λ2)を返す。値が大きいほど細長い。
    """
    if coords_xy.shape[0] < 2:
        return 0.0

    # 平均0にして共分散
    centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)

    # 数値不安定対策
    if not np.all(np.isfinite(cov)):
        return 0.0

    vals, _ = np.linalg.eig(cov)
    vals = np.sort(np.abs(vals))[::-1]  # 大きい方から

    if vals[1] <= 1e-6:
        return float("inf")

    return float(vals[0] / vals[1])


def main():
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"input not found: {INPUT}"

    binary = (img > 0).astype(np.uint8)

    # ラベリング
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)

    print(f"components: {len(props)}")

    output = np.zeros_like(binary)

    kept = 0
    for prop in props:
        coords_rc = prop.coords  # (N, 2) [row, col]
        n = coords_rc.shape[0]

        if n < MIN_POINTS:
            continue

        # shapely 用に (x, y) = (col, row)
        coords_xy = np.stack([coords_rc[:, 1], coords_rc[:, 0]], axis=1)
        line = LineString(coords_xy)

        length = line.length
        if length < MIN_LENGTH:
            continue

        elong = compute_elongation(coords_xy)
        if elong < MIN_ELONGATION:
            continue

        # 採用
        kept += 1
        output[coords_rc[:, 0], coords_rc[:, 1]] = 255

    print(f"kept components: {kept}")

    cv2.imwrite(OUTPUT, output)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
