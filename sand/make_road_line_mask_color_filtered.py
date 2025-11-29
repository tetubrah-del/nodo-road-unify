# make_road_line_mask_color_filtered.py
import cv2
import numpy as np

AERIAL = "data/aerial_tsujiwara_1km_z17.jpg"
OUT_FOREST_MASK = "forest_mask_color.png"
OUT_NOFOREST = "aerial_no_forest_for_road.png"
OUT_LINE_MASK = "road_line_mask_color_filtered.png"

def main():
    img = cv2.imread(AERIAL)
    if img is None:
        raise RuntimeError(f"failed to read {AERIAL}")
    h, w, _ = img.shape

    # --- 1) HSV で「森（濃い緑）」を取る ----------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # OpenCV の H は 0–179。だいたい緑っぽい範囲を広めに取る
    lower_green = np.array([35, 40, 30], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # すごく暗いところ（川・影・既に真っ黒な部分）も「森と同じ扱い」で消してしまう
    v = hsv[:, :, 2]
    dark_mask = (v < 40).astype(np.uint8) * 255

    forest_mask = cv2.bitwise_or(green_mask, dark_mask)
    cv2.imwrite(OUT_FOREST_MASK, forest_mask)

    # --- 2) 森＋暗部を真っ黒にした画像を作る -------------------
    img_no_forest = img.copy()
    img_no_forest[forest_mask > 0] = (0, 0, 0)
    cv2.imwrite(OUT_NOFOREST, img_no_forest)

    # --- 3) この画像から道路っぽい線だけ抜く -------------------
    gray = cv2.cvtColor(img_no_forest, cv2.COLOR_BGR2GRAY)

    # コントラスト強調（道路の明るい帯を浮かせる）
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    # トップハットで「細い明るい帯」を強調（道路＋畦道）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, kernel)

    # Canny でエッジ抽出
    edges = cv2.Canny(tophat, 40, 120)

    # 森マスクの中は強制的に 0 にしてエッジを消す（境目も含めて無視）
    edges[forest_mask > 0] = 0

    # スケルトン化（1ピクセル幅に）
    skel = cv2.ximgproc.thinning(edges)

    cv2.imwrite(OUT_LINE_MASK, skel)
    print("saved:", OUT_LINE_MASK)


if __name__ == "__main__":
    main()
