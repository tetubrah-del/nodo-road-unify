import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

INPUT = "road_line_mask_skel.png"
OUTPUT = "road_line_mask_skel_noloop.png"

# ★ パラメータ（あなたの要望）
CURV_WINDOW = 600   # 曲率を測る窓幅(px)
CURV_THRESHOLD = 2.5  # これ以上曲がっていれば「クルン」とみなす

def compute_curvature(points, window):
    """点列の局所曲率を推定"""
    if len(points) < window * 2 + 1:
        return 0

    pts = np.array(points, dtype=float)

    # 3点を等間隔にサンプリングして角度を求める
    p1 = pts[0]
    p2 = pts[len(pts)//2]
    p3 = pts[-1]

    v1 = p2 - p1
    v2 = p3 - p2

    def angle(v1, v2):
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
        return np.arccos(np.clip(cosang, -1.0, 1.0))

    return angle(v1, v2)

def main():
    img = cv2.imread(INPUT, 0)
    assert img is not None, f"not found: {INPUT}"

    # 二値化
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # ラベリング
    lbl = label(bw > 0)
    props = regionprops(lbl)

    cleaned = np.zeros_like(bw)

    for region in props:
        coords = region.coords

        # 曲率を計算
        curv = compute_curvature(coords, CURV_WINDOW)

        # ★ クルン判定 → 削除
        if curv > CURV_THRESHOLD:
            continue

        # 保存
        for (y, x) in coords:
            cleaned[y, x] = 255

    cv2.imwrite(OUTPUT, cleaned)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
