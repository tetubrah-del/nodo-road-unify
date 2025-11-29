# forest_detect_texture.py
import cv2
import numpy as np

INPUT = "data/aerial_tsujiwara_1km_z17.jpg"
OUTPUT = "forest_mask_texture.png"

# --- パラメータ ---
WIN = 7               # 7x7 の局所領域で分散を見る
VAR_THRESHOLD = 35.0  # 分散の閾値（大→森）

def main():
    # 画像を読み取り
    img = cv2.imread(INPUT)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 局所平均
    mean = cv2.blur(gray.astype(np.float32), (WIN, WIN))

    # 局所２乗平均
    mean2 = cv2.blur((gray.astype(np.float32) ** 2), (WIN, WIN))

    # 分散 = E[x^2] - (E[x])^2
    var = mean2 - mean ** 2

    # 分散が高い → 森
    forest_mask = (var > VAR_THRESHOLD).astype(np.uint8) * 255

    cv2.imwrite(OUTPUT, forest_mask)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
