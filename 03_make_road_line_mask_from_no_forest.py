import cv2
import numpy as np
from skimage.morphology import skeletonize
import argparse
import os

def run(input_path, output_path):
    print(f"[INFO] loading: {input_path}")
    img = cv2.imread(input_path)

    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny
    edges = cv2.Canny(gray, 80, 200)

    # 細線化
    skel = skeletonize(edges > 0).astype(np.uint8) * 255

    # 保存
    cv2.imwrite(output_path, skel)
    print(f"[INFO] saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input PNG")
    parser.add_argument("--output", required=True, help="output skeleton PNG")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
