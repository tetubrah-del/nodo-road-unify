# filter_road_skeleton_components.py
import cv2
import numpy as np

INPUT  = "road_skel_no_forest.png"
OUTPUT = "road_skel_no_forest_filtered.png"

# パラメータ
MIN_LEN_PX     = 60    # バウンディングボックスの長辺がこれ未満なら捨てる
MIN_AREA_PX    = 40    # ごく小さいゴミ除去
MAX_DENSITY    = 0.25  # area / (w*h) がこれより大きい＝ぐしゃっと詰まってる＝森・田んぼ枠とみなす

def main():
    skel = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        raise RuntimeError(f"failed to read {INPUT}")

    binary = (skel > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print("num_labels:", num_labels)

    out = np.zeros_like(binary)
    kept = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < MIN_AREA_PX:
            continue

        long_side = max(w, h)
        if long_side < MIN_LEN_PX:
            # 短すぎる線 → 小さな森の輪郭/ノイズ扱い
            continue

        bbox_area = w * h
        if bbox_area <= 0:
            continue

        density = float(area) / float(bbox_area)

        # density が高いほど「ボックスの中に線がぎっしり詰まってる」＝クルン系
        if density > MAX_DENSITY:
            continue

        out[labels == i] = 1
        kept += 1

    print(f"kept components: {kept}")

    out_u8 = (out * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out_u8)
    print("saved:", OUTPUT)

if __name__ == "__main__":
    main()
