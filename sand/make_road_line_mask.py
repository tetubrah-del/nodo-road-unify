import cv2
import numpy as np

INPUT = "data/aerial_no_veg.png"   # ← ダブルクォーテーション閉じ忘れ修正
OUTPUT = "road_line_mask.png"

def main():
    img = cv2.imread(INPUT)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge detect
    edges = cv2.Canny(gray, 60, 150)

    # 細線化（OpenCV版）
    skeleton = cv2.ximgproc.thinning(edges)

    cv2.imwrite(OUTPUT, skeleton)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
