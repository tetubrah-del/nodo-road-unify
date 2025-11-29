import cv2
import numpy as np
from pathlib import Path

IMAGE_PATH = "data/aerial_tsujiwara_10km_approx.jpg"
OUTPUT_MASK = "road_mask_color_kmeans.png"

# K-means のクラスタ数
K = 4

def main():
    path = Path(IMAGE_PATH)
    assert path.exists(), f"input not found: {path}"

    # 1. 画像読み込み（BGR）
    bgr = cv2.imread(str(path))
    h, w, _ = bgr.shape
    print("input shape:", bgr.shape)

    # 2. Lab 色空間に変換（明るさと色差が分かりやすい）
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lab_f = lab.reshape(-1, 3).astype(np.float32)

    # 3. K-means クラスタリング
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        lab_f,
        K,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.astype(np.float32)
    print("centers (L,a,b):")
    print(centers)

    # 4. 「明るくて無彩色寄り」のクラスタを選ぶ
    #    → L が大きく、|a|+|b| が小さいクラスタ
    L = centers[:, 0]
    chroma = np.abs(centers[:, 1]) + np.abs(centers[:, 2])

    # スコア = L - α * chroma
    alpha = 0.5
    scores = L - alpha * chroma
    best_idx = int(np.argmax(scores))
    print("selected cluster index:", best_idx)

    road_mask_flat = (labels.flatten() == best_idx).astype(np.uint8) * 255
    road_mask = road_mask_flat.reshape(h, w)

    # 5. ノイズ除去（少し膨張 → 収縮）
    kernel = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite(OUTPUT_MASK, road_mask)
    print("saved:", OUTPUT_MASK)


if __name__ == "__main__":
    main()
