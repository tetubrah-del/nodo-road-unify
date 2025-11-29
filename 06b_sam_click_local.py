#!/usr/bin/env python
"""
06b_sam_click_local.py

1km タイルなど「広い画像」用のクリック式 SAM。

- クリックした点の“周辺だけ”を対象に SAM を実行する
  （クリック群のバウンディングボックス＋マージンで box 制約）

- 操作
    左クリック : 農道の上を数点クリック
    's'         : そのクリック群の周辺だけ SAM → クリックとつながる部分だけ global_mask に追加
                  → その場でマスク＆オーバーレイを保存
    'q' / ESC   : 終了（保存済み）

使用例:

  python 06b_sam_click_local.py \
    --image-path data/aerial_tsujiwara_1km_z17.jpg \
    --checkpoint data/sam_vit_h.pth \
    --model-type vit_h \
    --output-mask data/tsuji_1km_roads_mask_local.png \
    --output-overlay data/tsuji_1km_roads_overlay_local.png
"""

import argparse
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


# ------------------------------------------------------------
# 引数
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--output-mask", default="sam_roads_mask_local.png")
    p.add_argument("--output-overlay", default="sam_roads_overlay_local.png")
    p.add_argument(
        "--margin",
        type=int,
        default=128,
        help="クリック点周辺に付けるバウンディングボックスのマージン（ピクセル）",
    )
    return p.parse_args()


# ------------------------------------------------------------
# （今は未使用だが将来用の HSV / 細長フィルタ）
# ------------------------------------------------------------

def refine_mask_with_hsv(
    img_bgr: np.ndarray,
    sam_mask: np.ndarray,
    sat_thresh: int = 80,
    val_min: int = 120,
    val_max: int = 255,
) -> np.ndarray:
    """
    SAM マスクのうち、HSV 的に「グレー寄りでそこそこ明るい」画素だけ残す。
    道路色っぽくない田んぼの領域を削るためのフィルタ。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    grayish = (s < sat_thresh)
    bright = (v > val_min) & (v < val_max)

    hsv_mask = np.zeros_like(s, dtype=np.uint8)
    hsv_mask[grayish & bright] = 255

    combined = cv2.bitwise_and(sam_mask, hsv_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return combined


def filter_slender_components(
    mask: np.ndarray,
    min_area: int = 200,
    max_area: int = 8000,
    min_aspect: float = 3.0,
) -> np.ndarray:
    """
    面積とアスペクト比で「細長い成分」だけ残すフィルタ。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    out = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):  # 0 は背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect < min_aspect:
            continue

        out[labels == i] = 255

    return out


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

def main():
    args = parse_args()

    # -------- 画像ロード --------
    print(f"[INFO] Loading image: {args.image_path}")
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(args.image_path)
    h, w = img_bgr.shape[:2]
    print(f"[INFO] Image size: {w} x {h}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # -------- SAM ロード --------
    print("[INFO] Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(img_rgb)

    # クリック点とグローバルマスク
    click_points = []
    global_mask = np.zeros((h, w), dtype=np.uint8)  # 0/255

    # -------- 出力保存関数 --------
    def save_outputs():
        if np.count_nonzero(global_mask) == 0:
            print("[INFO] global_mask is empty, nothing to save yet.")
            return

        cv2.imwrite(args.output_mask, global_mask)
        print(f"[INFO] Saved mask image -> {args.output_mask}")

        overlay = img_bgr.copy()
        ys, xs = np.where(global_mask > 0)
        if len(xs) > 0:
            color = np.array([0, 0, 255], dtype=np.uint8)
            alpha = 0.5
            overlay[ys, xs] = (
                alpha * color + (1 - alpha) * overlay[ys, xs]
            ).astype(np.uint8)

        cv2.imwrite(args.output_overlay, overlay)
        print(f"[INFO] Saved overlay image -> {args.output_overlay}")

    # -------- 表示更新 --------
    def redraw_overlay():
        overlay = img_bgr.copy()

        ys, xs = np.where(global_mask > 0)
        if len(xs) > 0:
            color = np.array([0, 0, 255], dtype=np.uint8)
            alpha = 0.5
            overlay[ys, xs] = (
                alpha * color + (1 - alpha) * overlay[ys, xs]
            ).astype(np.uint8)

        for (cx, cy) in click_points:
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow("image", overlay)

    # -------- マウスコールバック --------
    def on_mouse(event, x, y, flags, param):
        nonlocal click_points
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append((x, y))
            print(f"[INFO] Clicked: ({x}, {y})")
            redraw_overlay()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_mouse)
    redraw_overlay()

    print("[INFO] 操作方法:")
    print("  - 農道の上を左クリック（1〜数点）")
    print("  - 's' でそのクリック群の“周辺だけ” SAM → マスク追加＆即保存")
    print("  - 'q' or ESC で終了（いつでも Ctrl+C で落としても、直前の結果は保存済）")

    # -------- メインループ --------
    while True:
        key = cv2.waitKey(10) & 0xFF

        if key != 255:
            print(f"[DEBUG] key code: {key}")

        # 終了
        if key in (ord("q"), 27):  # 'q' or ESC
            print("[INFO] Quit.")
            break

        # セグメント
        if key == ord("s"):
            if not click_points:
                print("[WARN] クリック点がありません。")
                continue

            # ---- クリック群のローカル領域を決める ----
            xs = [p[0] for p in click_points]
            ys = [p[1] for p in click_points]
            margin = args.margin

            x_min = max(0, min(xs) - margin)
            x_max = min(w - 1, max(xs) + margin)
            y_min = max(0, min(ys) - margin)
            y_max = min(h - 1, max(ys) + margin)

            box = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
            print(f"[INFO] Local box: ({x_min}, {y_min}) - ({x_max}, {y_max})")

            # ---- SAM 実行（ローカル box 制約付き）----
            input_points = np.array(click_points, dtype=np.float32)
            input_labels = np.ones(len(click_points), dtype=np.int32)

            print(f"[INFO] SAM prediction with {len(click_points)} points (local box)...")
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=box,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(np.uint8)
            print(f"[INFO] SAM mask score: {scores[best_idx]:.3f}")

            raw_mask = (best_mask > 0).astype(np.uint8) * 255

            # box の外をゼロに（保険）
            raw_mask[:y_min, :] = 0
            raw_mask[y_max + 1 :, :] = 0
            raw_mask[:, :x_min] = 0
            raw_mask[:, x_max + 1 :] = 0

            print(f"[INFO] Raw SAM pixels (in box): {int(np.count_nonzero(raw_mask))}")

            # ---- ゆるいモルフォロジーだけかける ----
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            if np.count_nonzero(refined) == 0:
                print("[WARN] フィルタ後マスクが空です。このクリック群は無視します。")
                click_points = []
                redraw_overlay()
                continue

            # ---- クリックと同じ連結成分だけ残す ----
            num_labels, labels = cv2.connectedComponents(
                (refined > 0).astype(np.uint8),
                connectivity=8,
            )

            labels_to_keep = set()
            for (cx, cy) in click_points:
                if 0 <= cx < w and 0 <= cy < h:
                    lab = labels[cy, cx]
                    if lab != 0:
                        labels_to_keep.add(lab)

            if not labels_to_keep:
                print("[WARN] クリック位置に属するラベルが見つからなかったので、このクリック群は無視します。")
                click_points = []
                redraw_overlay()
                continue

            final = np.zeros_like(refined, dtype=np.uint8)
            for lab in labels_to_keep:
                final[labels == lab] = 255

            nonzero_final = int(np.count_nonzero(final))
            print(f"[INFO] Final pixels (connected to clicks): {nonzero_final}")

            if nonzero_final == 0:
                print("[WARN] 最終マスクが空です。")
                click_points = []
                redraw_overlay()
                continue

            # ---- global マスクに追加して即保存 ----
            global_mask = cv2.bitwise_or(global_mask, final)
            print(
                f"[INFO] Global mask total nonzero: "
                f"{int(np.count_nonzero(global_mask))}"
            )
            save_outputs()

            # クリック点クリア
            click_points = []
            redraw_overlay()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
