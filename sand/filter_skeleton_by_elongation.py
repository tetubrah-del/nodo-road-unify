# filter_skeleton_by_elongation.py
import cv2
import numpy as np

INPUT  = "road_line_mask_skel.png"          # ここを手元のベストな skeleton に
OUTPUT = "road_line_mask_skel_elong.png"

# パラメータ
MIN_PIXELS       = 30    # あまりに小さい成分は最初からノイズ扱いで捨てる
ELONGATION_TH    = 0.85  # どれくらい「細長ければ道路」とみなすか（0.8〜0.9で調整）
MIN_KEEP_LENGTH  = 40    # 残す道路の最低ピクセル数（短すぎるのは消す）

def main():
    img = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 0,1 に正規化
    binary = (img > 0).astype(np.uint8)

    # ラベリング
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    out = np.zeros_like(binary)
    kept = 0
    removed = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_PIXELS:
            removed += 1
            continue

        # この成分の座標を取得
        ys, xs = np.where(labels == label)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)

        # 長さ（ピクセル数）
        length = coords.shape[0]
        if length < MIN_KEEP_LENGTH:
            removed += 1
            continue

        # 共分散行列 → 固有値（主成分）
        cov = np.cov(coords.T)
        eigvals, _ = np.linalg.eig(cov)
        eigvals = np.sort(eigvals)[::-1]  # 大きい順

        # 数値誤差対策
        if eigvals[0] <= 0:
            removed += 1
            continue

        elong_ratio = float(eigvals[0] / (eigvals[0] + eigvals[1]))

        # 細長さで判定
        if elong_ratio >= ELONGATION_TH:
            out[labels == label] = 1
            kept += 1
        else:
            removed += 1

    print(f"kept components   : {kept}")
    print(f"removed components: {removed}")

    # 画像として保存
    out_img = (out * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out_img)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
