# filter_skeleton_by_color.py
import cv2
import numpy as np

AERIAL_PATH = "data/aerial_tsujiwara_1km_z17.jpg"   # 航空写真
SKEL_PATH   = "road_line_mask_skel.png"             # いちばん精度良さそうなスケルトン
OUTPUT      = "road_line_mask_color_filtered.png"

# ===== パラメータ（ここをいじって調整） =====
# HSV で「濃い緑」とみなす範囲
GREEN_H_MIN = 35   # 0-179
GREEN_H_MAX = 85
SAT_MIN     = 60   # 彩度の下限（0-255）
VAL_MIN     = 40   # 明度の下限（真っ暗な影を除く）

# Excess Green の閾値（2G - R - B がこれより大きいと「緑優勢」とみなす）
EXG_THRESH  = 20

# 森林マスクをどれくらい膨張させるか（道路境界までまとめて削る用）
DILATE_KERNEL = 3    # 奇数。3 か 5 くらいがおすすめ

def main():
    # 画像読み込み
    aerial = cv2.imread(AERIAL_PATH)
    if aerial is None:
        raise RuntimeError(f"failed to read aerial: {AERIAL_PATH}")

    skel = cv2.imread(SKEL_PATH, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        raise RuntimeError(f"failed to read skeleton: {SKEL_PATH}")

    # サイズが違う場合は、スケルトンに合わせて航空写真をリサイズ
    h_s, w_s = skel.shape
    h_a, w_a, _ = aerial.shape
    if (h_s, w_s) != (h_a, w_a):
        print(f"resize aerial from {(h_a, w_a)} to {(h_s, w_s)}")
        aerial = cv2.resize(aerial, (w_s, h_s), interpolation=cv2.INTER_LINEAR)

    # BGR -> HSV
    hsv = cv2.cvtColor(aerial, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # BGR チャンネルも取り出して Excess Green を計算
    b, g, r = cv2.split(aerial.astype(np.int16))
    exg = 2 * g - r - b   # 負の値もあるので int16 のまま

    # 「緑優勢」かどうか（Excess Green）
    exg_mask = (exg > EXG_THRESH)

    # HSV で「緑っぽい」「そこそこ明るくて彩度がある」ピクセル
    green_hsv_mask = (
        (h >= GREEN_H_MIN) & (h <= GREEN_H_MAX) &
        (s >= SAT_MIN) & (v >= VAL_MIN)
    )

    # 上の両方を満たすところを「森林」とみなす
    forest_mask = (exg_mask & green_hsv_mask).astype(np.uint8)

    # 森林マスクを膨張して、境界線も含めて広げる
    if DILATE_KERNEL > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL)
        )
        forest_mask = cv2.dilate(forest_mask, k, iterations=1)

    # デバッグ用に森林マスクも保存しておく（真っ白が森）
    cv2.imwrite("forest_mask_debug.png", forest_mask * 255)

    # スケルトンを 0/1 に
    skel_bin = (skel > 0).astype(np.uint8)

    # 「森林ではない場所」のマスク
    non_forest_mask = (forest_mask == 0).astype(np.uint8)

    # スケルトン＆非森林 → 森の上の線だけ削除
    filtered = cv2.bitwise_and(skel_bin, skel_bin, mask=non_forest_mask)

    kept_ratio = filtered.sum() / (skel_bin.sum() + 1e-6)
    print(f"kept skeleton pixels ratio: {kept_ratio:.3f}")

    out = (filtered * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
