import cv2
import numpy as np
import math

INPUT = "road_line_mask.png"
OUTPUT = "road_extracted_contour.png"

def calc_curvature(cnt):
    pts = cnt.reshape(-1, 2)
    # 曲率 = 3点の角度の平均を使う簡易スコア
    curv = 0
    for i in range(1, len(pts) - 1):
        a = pts[i - 1]
        b = pts[i]
        c = pts[i + 1]
        v1 = a - b
        v2 = c - b
        ang = math.acos(
            max(-1, min(1, np.dot(v1, v2) / 
                (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)))
        )
        curv += ang
    return curv / len(pts)

def main():
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cannot read input")

    # 0/255 → 0/1
    bin_img = (img > 0).astype(np.uint8)

    # 輪郭抽出
    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    result = np.zeros_like(bin_img)

    for cnt in contours:
        # 閉じてる？
        is_closed = cv2.isContourConvex(cnt)

        # 面積
        area = cv2.contourArea(cnt)

        # 周囲長
        length = cv2.arcLength(cnt, False)

        # バウンディング長方形
        x, y, w, h = cv2.boundingRect(cnt)
        elong = max(w, h) / (min(w, h) + 1e-6)

        # 曲率
        curvature = calc_curvature(cnt)

        # --- 農道フィルタ条件 ---
        if (
            length > 80 and          # そこそこ長い
            elong > 3 and            # 細長い
            area < 2000 and          # 田んぼ除外
            curvature < 2.0 and      # ぐにゃぐにゃ排除
            (not is_closed)          # 田んぼの枠を除外
        ):
            cv2.drawContours(result, [cnt], -1, 1, 1)

    # 保存
    cv2.imwrite(OUTPUT, result * 255)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
