# make_road_skeleton_from_no_forest.py
import cv2
import numpy as np
from skimage.morphology import skeletonize

INPUT  = "data/aerial_no_forest_v2.png"   # パスは環境に合わせて
EDGE_OUT = "road_edges_no_forest.png"
SKEL_OUT = "road_skel_no_forest.png"

def main():
    img = cv2.imread(INPUT)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ノイズを落として細い線を強調
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny。森を黒く潰しているので少し弱めのしきい値に
    edges = cv2.Canny(blur, threshold1=30, threshold2=90)
    cv2.imwrite(EDGE_OUT, edges)
    print("saved:", EDGE_OUT)

    # ちぎれたエッジを少し繋ぐ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 0/1 にしてスケルトン化
    binary = (closed > 0).astype(np.uint8)
    skel = skeletonize(binary > 0)
    skel_u8 = (skel.astype(np.uint8) * 255)

    cv2.imwrite(SKEL_OUT, skel_u8)
    print("saved:", SKEL_OUT)

if __name__ == "__main__":
    main()
