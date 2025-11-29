# remove_forest_v3.py
import cv2
import numpy as np

INPUT  = "data/aerial_tsujiwara_1km_z17.jpg"
OUTPUT = "aerial_no_forest_v3.png"

def main():
    img = cv2.imread(INPUT)
    if img is None:
        raise RuntimeError("FAILED TO READ input image")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 森の緑色（低彩度〜高彩度まで幅広く拾う）
    lower = np.array([20,  20, 20])   # H:20 付近から
    upper = np.array([90, 255,255])   # H:90 付近まで

    mask = cv2.inRange(hsv, lower, upper)

    # ❗森と思われる部分を完全に黒に塗りつぶす
    img[mask > 0] = (0,0,0)

    cv2.imwrite(OUTPUT, img)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
