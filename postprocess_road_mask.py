import cv2
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path

INPUT_MASK = "road_mask_tsujiwara.png"
OUTPUT_SKELETON = "road_mask_tsujiwara_skel.png"

# この長さ未満の線はノイズとして削除（適宜調整）
MIN_COMPONENT_LENGTH = 100   # ピクセル

def main():
    path = Path(INPUT_MASK)
    assert path.exists(), f"mask not found: {path}"

    # 1. マスク読み込み（グレースケール）
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # 2. 二値化（閾値は適宜調整）
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ※ もし「道路が黒、背景が白」になっている場合は反転
    # 白=道路, 黒=背景 に統一したい
    white_ratio = np.mean(bin_mask == 255)
    if white_ratio < 0.5:
        # 画面の大半が黒なら、多分「道路が黒」パターンなので反転
        bin_mask = cv2.bitwise_not(bin_mask)

    # 3. 軽くノイズ除去（オープニング）
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. 細線化（スケルトン化）
    # ximgproc.thinning を使う
    skeleton = cv2.ximgproc.thinning(clean)

    # 5. 連結成分ラベリングして、短い線を削除
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8
    )

    # stats: [label, x, y, w, h, area]
    result = np.zeros_like(skeleton)

    for label in range(1, num_labels):  # 0 は背景
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_LENGTH:
            result[labels == label] = 255

    # 6. 保存
    cv2.imwrite(OUTPUT_SKELETON, result)
    print("Saved skeletonized mask to:", OUTPUT_SKELETON)


if __name__ == "__main__":
    main()
