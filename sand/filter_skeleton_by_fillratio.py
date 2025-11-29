# filter_skeleton_by_fillratio.py
import cv2
import numpy as np

INPUT  = "road_line_mask_skel.png"          # さっき「一番良い」と言っていた skeleton 版
OUTPUT = "road_line_mask_skel_fillratio.png"

# パラメータ
MIN_PIXELS  = 40    # これ未満の小さい成分はノイズ扱いで捨てる
MIN_RATIO   = 0.35  # 塗りつぶし率の下限。小さいほどクルンも残りやすくなる
MIN_WIDTH   = 3     # ごく薄い箱（h=1,w=○○など）だけの変な成分を避けるための保険
MIN_HEIGHT  = 3

def main():
    skel = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 0/1 バイナリ化
    binary = (skel > 0).astype(np.uint8)

    # 連結成分解析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # stats: [label, x, y, w, h, area_px]

    kept = 0
    out = np.zeros_like(binary, dtype=np.uint8)

    for i in range(1, num_labels):  # 0 は背景
        x, y, w, h, area_px = stats[i]

        # ごく小さい成分はノイズとして除外
        if area_px < MIN_PIXELS:
            continue

        # ごく小さい bbox は（点の塊っぽいので）除外
        if w < MIN_WIDTH and h < MIN_HEIGHT:
            continue

        bbox_area = w * h
        if bbox_area == 0:
            continue

        fill_ratio = area_px / float(bbox_area)

        # 「細長くて、箱をそこそこ埋めている線」だけ残す
        if fill_ratio >= MIN_RATIO:
            out[labels == i] = 1
            kept += 1

    print(f"num_labels : {num_labels}")
    print(f"kept comps : {kept}")

    out_img = (out * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out_img)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
