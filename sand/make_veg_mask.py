# make_veg_mask.py
import cv2
import numpy as np

INPUT  = "data/aerial_tsujiwara_1km_z17.jpg"
MASK_OUT = "veg_mask.png"
AERIAL_NO_VEG = "aerial_no_veg.png"

# パラメータ
NDVI_MIN = 0.08   # ここより大きいと「植生っぽい」
MIN_PATCH = 200   # 小さすぎる緑パッチは無視（ノイズ除去用）

def main():
    img_bgr = cv2.imread(INPUT)
    if img_bgr is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # BGR → float32, チャンネル分離
    b, g, r = cv2.split(img_bgr.astype(np.float32))

    # 疑似 NDVI = (G - R) / (G + R)
    ndvi = (g - r) / (g + r + 1e-6)

    # 値の範囲はだいたい -1〜1 になる想定
    veg_mask = (ndvi > NDVI_MIN).astype(np.uint8)

    # ついでに「暗いところ（川・影など）」も森扱いして消したいなら、
    # HSV の V が低いところを追加でマスクしてもいい
    hsv = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_mask = (v < 40).astype(np.uint8)

    # 植生 or 真っ暗な場所 → 1
    veg_mask = np.clip(veg_mask + dark_mask, 0, 1)

    # 小さなパッチを削除（つぶつぶ対策）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        veg_mask, connectivity=8
    )
    cleaned = np.zeros_like(veg_mask)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_PATCH:
            continue
        cleaned[labels == i] = 1
        kept += 1
    print(f"vegetation components kept: {kept}")

    veg_mask_u8 = (cleaned * 255).astype(np.uint8)
    cv2.imwrite(MASK_OUT, veg_mask_u8)
    print("saved:", MASK_OUT)

    # 植生＋暗部を黒で塗りつぶした航空写真
    aerial_no_veg = img_bgr.copy()
    aerial_no_veg[cleaned == 1] = (0, 0, 0)
    cv2.imwrite(AERIAL_NO_VEG, aerial_no_veg)
    print("saved:", AERIAL_NO_VEG)


if __name__ == "__main__":
    main()
