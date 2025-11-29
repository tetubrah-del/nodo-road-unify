# filter_skeleton_by_length.py
import cv2
import numpy as np

INPUT  = "road_line_mask_skel.png"          # さっき「いいね！」と言ってたやつ
OUTPUT = "road_line_mask_skel_len150.png"   # 出力ファイル名

MIN_LENGTH = 150   # ★ここがしきい値。あとで 100, 200 に変えて実験もOK

def main():
    skel = cv2.imread(INPUT, cv2.IMREAD_GRAYSCALE)
    if skel is None:
        raise RuntimeError(f"failed to read {INPUT}")

    # 0/1 に変換
    binary = (skel > 0).astype(np.uint8)

    # ラベリング（連結成分ごとに分ける）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    print("num_labels:", num_labels)

    # 出力用キャンバス
    out = np.zeros_like(binary)

    kept = 0
    for i in range(1, num_labels):   # 0 は背景なので飛ばす
        area = stats[i, cv2.CC_STAT_AREA]  # 細線なので area ≒ 長さ(px)

        if area < MIN_LENGTH:
            # 短すぎる線は捨てる
            continue

        # 条件を満たした線だけ残す
        out[labels == i] = 1
        kept += 1

    print(f"kept components: {kept}")

    out_img = (out * 255).astype(np.uint8)
    cv2.imwrite(OUTPUT, out_img)
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
