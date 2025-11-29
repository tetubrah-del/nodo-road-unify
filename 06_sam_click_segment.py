#!/usr/bin/env python
"""
クリック式 SAM で農道マスクを累積して作るシンプル版。

- 左クリック: 農道の上を数点クリック
- 's'        : いまのクリック群で SAM 実行 → クリックと同じ連結成分だけ global_mask に追加
               → その場でマスク＆オーバーレイを保存
- 'q' / ESC  : 終了（保存はすでに終わっているので安心）
"""

import argparse
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--output-mask", default="sam_roads_mask_click.png")
    p.add_argument("--output-overlay", default="sam_roads_overlay_click.png")
    return p.parse_args()


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

    def save_outputs():
        """global_mask を画像として保存"""
        if np.count_nonzero(global_mask) == 0:
            print("[INFO] global_mask is empty, nothing to save yet.")
            return

        cv2.imwrite(args.output_mask, global_mask)
        print(f"[INFO] Saved mask image -> {args.output_mask}")

        overlay = img_bgr.copy()
        ys, xs = np.where(global_mask > 0)
        color = np.array([0, 0, 255], dtype=np.uint8)
        alpha = 0.5
        overlay[ys, xs] = (
            alpha * color + (1 - alpha) * overlay[ys, xs]
        ).astype(np.uint8)
        cv2.imwrite(args.output_overlay, overlay)
        print(f"[INFO] Saved overlay image -> {args.output_overlay}")

    def redraw_overlay():
        """現在の global_mask とクリック点を表示"""
        overlay = img_bgr.copy()
        ys, xs = np.where(global_mask > 0)
        if len(xs) > 0:
            color = np.array([0, 0, 255], dtype=np.uint8)
            alpha = 0.5
            overlay[ys, xs] = (
                alpha * color + (1 - alpha) * overlay[ys, xs]
            ).astype(np.uint8)
        # クリック中の点
        for (cx, cy) in click_points:
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
        cv2.imshow("image", overlay)

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
    print("  - 's' でその道路をセグメントしてマスクに追加＆即保存")
    print("  - 'q' or ESC で終了（いつでも Ctrl+C で落としても、直前の結果は保存済）")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key != 255:
            print(f"[DEBUG] key code: {key}")

        if key in (ord("q"), 27):  # 'q' or ESC
            print("[INFO] Quit.")
            break

        if key == ord("s"):
            if not click_points:
                print("[WARN] クリック点がありません。")
                continue

            # -------- SAM 実行（いまのクリック群） --------
            input_points = np.array(click_points, dtype=np.float32)
            input_labels = np.ones(len(click_points), dtype=np.int32)

            print(f"[INFO] SAM prediction with {len(click_points)} points...")
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(np.uint8)
            print(f"[INFO] SAM mask score: {scores[best_idx]:.3f}")

            raw_mask = (best_mask > 0).astype(np.uint8) * 255
            print(f"[INFO] Raw SAM pixels: {int(np.count_nonzero(raw_mask))}")

            # 軽くオープン（ノイズ削り）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # ---- クリックと同じ連結成分だけ残す ----
            num_labels, labels = cv2.connectedComponents(
                (refined > 0).astype(np.uint8),
                connectivity=8,
            )

            labels_to_keep = set()
            for (x, y) in click_points:
                if 0 <= x < w and 0 <= y < h:
                    lab = labels[y, x]
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

            # global マスクに追加して即保存
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
