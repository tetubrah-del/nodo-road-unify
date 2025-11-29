# 02_remove_forest_color_v3.py

import cv2
import numpy as np
import argparse


def remove_forest(input_path: str, output_path: str):
    print(f"[INFO] loading: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 「緑っぽい」領域をざっくりマスク（前に使っていた設定でOK）
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # マスクを膨張・収縮して少し滑らかに
    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # 森エリアを「明るい色」にして道路を強調
    result = img.copy()
    result[mask_green > 0] = (200, 200, 200)

    cv2.imwrite(output_path, result)
    print(f"[INFO] saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="元の航空写真パス")
    parser.add_argument("--output", required=True, help="森除去後の画像出力パス")
    args = parser.parse_args()

    remove_forest(args.input, args.output)


if __name__ == "__main__":
    main()
