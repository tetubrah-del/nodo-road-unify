# filter_loopy_roads.py
#
# road_line_mask_skel_v3.png の中から
# 「丸っこい / くるんとした」成分を除外して、
# 細長い道路っぽい成分だけ残すフィルタ

import cv2
import numpy as np

# ===== パラメータ =====
INPUT  = "road_line_mask_skel_v3.png"          # 元のスケルトン画像
OUTPUT = "road_line_mask_skel_v3_clean.png"    # 出力ファイル名

MIN_AREA     = 30    # 小さすぎる成分はノイズとして捨てる
ANISO_MIN    = 4.0   # 伸び具合のしきい値（大きいほど「細長い」成分だけ残る）

def main():
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 0/1 に変換
    binary = (img > 0).astype(np.uint8)

    # 連結成分ラベリング
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    out = np.zeros_like(binary, dtype=np.uint8)

    kept = 0
    removed = 0

    for i in range(1, num_labels):  # 0 は背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            # 小さいノイズ
            removed += 1
            continue

        # この成分の座標を取得（y, x）
        ys, xs = np.where(labels == i)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)

        # 中心化
        center = coords.mean(axis=0, keepdims=True)
        coords_centered = coords - center

        # 共分散行列 → 固有値
        cov = (coords_centered.T @ coords_centered) / len(coords_centered)
        eigvals, _ = np.linalg.eig(cov)
        eigvals = np.sort(np.real(eigvals))[::-1]  # 大きい方, 小さい方

        # 伸び具合（異方性） = λmax / λmin
        # 丸っこい形 → ratio ~ 1
        # 細長い形 → ratio >> 1
        if eigvals[1] <= 0:
            ratio = np.inf
        else:
            ratio = float(eigvals[0] / eigvals[1])

        if ratio >= ANISO_MIN:
            # 細長い → 道路っぽい → 残す
            out[labels == i] = 1
            kept += 1
        else:
            # 丸い・ぐちゃっとした → 森のクルン・畑境界の輪郭 → 消す
            removed += 1

    print(f"components total: {num_labels - 1}")
    print(f"kept:    {kept}")
    print(f"removed: {removed}")

    out_img = (out * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out_img)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
