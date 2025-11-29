# filter_by_curvature.py

import cv2
import numpy as np
from skimage import measure, morphology
from skimage.io import imread, imsave
import math
from pathlib import Path

INPUT = "road_line_mask_skel.png"
OUTPUT = "road_line_mask_curvature_filtered.png"

def curvature(a, b, c):
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0
    cosv = np.clip(dot / denom, -1, 1)
    angle = np.arccos(cosv)
    return np.degrees(angle)

def calc_curvature_score(coords):
    if len(coords) < 5:
        return 999   # 曲率高い扱い

    curvs = []
    for i in range(1, len(coords)-1):
        a = coords[i-1]
        b = coords[i]
        c = coords[i+1]
        curvs.append(curvature(a, b, c))

    curvs = np.array(curvs)
    high_ratio = np.mean(curvs > 30)     # 30°以上の割合
    return high_ratio

def main():
    img = imread(INPUT, as_gray=True)
    img = (img > 0.5).astype(np.uint8)

    # connected components
    labeled = measure.label(img, connectivity=2)
    out = np.zeros_like(img)

    for label in range(1, labeled.max()+1):
        mask = (labeled == label)
        coords = np.column_stack(np.nonzero(mask))

        # 曲率ベース
        score = calc_curvature_score(coords)

        if score < 0.15:    # ★調整ポイント
            out[mask] = 1

    imsave(OUTPUT, out*255)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
