# analyze_component_widths.py
import cv2
import numpy as np
from collections import Counter

INPUT = "road_surface_mask.png"

def main():
    mask = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read {INPUT}")

    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )

    print("num_labels:", num_labels)

    width_hist = Counter()
    areas = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        short_side = min(w, h)
        width_hist[short_side] += 1
        areas.append(area)

    print("\n=== short_side histogram (px) ===")
    for width in sorted(width_hist.keys()):
        print(f"{width:2d}px : {width_hist[width]} components")

    if areas:
        areas = np.array(areas)
        print("\narea stats (px):")
        for q in [0, 25, 50, 75, 90, 95, 99, 100]:
            v = np.percentile(areas, q)
            print(f" {q:2d}th percentile: {v:.1f}")

if __name__ == "__main__":
    main()
