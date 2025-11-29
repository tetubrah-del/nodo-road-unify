#!/usr/bin/env python
"""
06c_manual_draw_roads.py

画像の上でクリックした点を線で結んで「手描き道路マスク」を作るツール。

操作:
  左クリック : 現在のポリラインに点を追加
  'n'         : 現在のポリラインを確定して次の道路へ（点が2つ以上あるとき）
  'u'         : 1つ前の点を取り消し（現在ポリライン）／ポリライン丸ごと削除
  's'         : 現在までのポリラインでマスク＆オーバーレイを保存
  'q' / ESC   : 終了（終了前にも保存する）

使い方:

  python 06c_manual_draw_roads.py \
    --image-path data/aerial_tsujiwara_1km_z17.jpg \
    --output-mask data/tsuji_1km_roads_mask_manual.png \
    --output-overlay data/tsuji_1km_roads_overlay_manual.png \
    --line-width 3
"""

import argparse
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True)
    p.add_argument("--output-mask", default="roads_mask_manual.png")
    p.add_argument("--output-overlay", default="roads_overlay_manual.png")
    p.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="描画する道路ラインの太さ（ピクセル）",
    )
    return p.parse_args()


def draw_polylines(mask, polylines, color=255, thickness=3):
    """polylines: [ [(x,y), (x,y), ...], ... ] を mask に描画"""
    for poly in polylines:
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            cv2.line(mask, poly[i], poly[i + 1], color, thickness=thickness)


def main():
    args = parse_args()

    print(f"[INFO] Loading image: {args.image_path}")
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(args.image_path)
    h, w = img_bgr.shape[:2]
    print(f"[INFO] Image size: {w} x {h}")

    # すべての道路ポリラインを格納するリスト
    polylines = []          # list[list[(x,y)]]
    current_poly = []       # いま編集中の1本
    line_width = args.line_width

    def build_overlay():
        """ベース画像 + 既存ポリライン + 編集中ポリラインを重ねた表示用画像を返す"""
        overlay = img_bgr.copy()

        # 既存ポリライン: 赤
        for poly in polylines:
            if len(poly) < 2:
                continue
            cv2.polylines(
                overlay,
                [np.array(poly, dtype=np.int32)],
                isClosed=False,
                color=(0, 0, 255),
                thickness=line_width,
            )

        # 編集中ポリライン: 黄
        if len(current_poly) >= 1:
            for pt in current_poly:
                cv2.circle(overlay, pt, 3, (0, 255, 255), -1)
        if len(current_poly) >= 2:
            cv2.polylines(
                overlay,
                [np.array(current_poly, dtype=np.int32)],
                isClosed=False,
                color=(0, 255, 255),
                thickness=line_width,
            )

        return overlay

    def redraw():
        cv2.imshow("image", build_overlay())

    def save_outputs():
        """現在の polylines からマスク＆オーバーレイを保存"""
        if not polylines:
            print("[INFO] No polylines yet, nothing to save.")
            return

        mask = np.zeros((h, w), dtype=np.uint8)
        draw_polylines(mask, polylines, color=255, thickness=line_width)
        cv2.imwrite(args.output_mask, mask)
        print(f"[INFO] Saved mask -> {args.output_mask}")

        overlay = img_bgr.copy()
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            color = np.array([0, 0, 255], dtype=np.uint8)
            alpha = 0.5
            overlay[ys, xs] = (
                alpha * color + (1 - alpha) * overlay[ys, xs]
            ).astype(np.uint8)
        cv2.imwrite(args.output_overlay, overlay)
        print(f"[INFO] Saved overlay -> {args.output_overlay}")

    # ---- マウスイベント ----
    def on_mouse(event, x, y, flags, param):
        nonlocal current_poly
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append((x, y))
            print(f"[INFO] Add point: ({x}, {y}) (current poly len={len(current_poly)})")
            redraw()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_mouse)
    redraw()

    print("[INFO] 操作方法:")
    print("  左クリック : 現在の道路の頂点を追加")
    print("  'n'        : 現在のポリラインを確定して次の道路へ（2点以上のとき）")
    print("  'u'        : 1点取り消し（現在のポリライン）。空なら最後の道路を削除")
    print("  's'        : 現在までの結果を保存（マスク & オーバーレイ）")
    print("  'q' / ESC  : 終了（終了前にも保存を試行）")

    while True:
        key = cv2.waitKey(10) & 0xFF

        if key != 255:
            print(f"[DEBUG] key code: {key}")

        if key in (ord("q"), 27):  # 'q' or ESC
            print("[INFO] Quit requested, saving before exit...")
            save_outputs()
            break

        # ポリライン確定
        if key == ord("n"):
            if len(current_poly) >= 2:
                polylines.append(current_poly.copy())
                print(f"[INFO] Commit polyline with {len(current_poly)} points. "
                      f"Total polylines: {len(polylines)}")
                current_poly = []
                redraw()
                # コミットのたびにセーブしてもいい
                save_outputs()
            else:
                print("[WARN] ポリラインが短すぎます（2点以上必要）。")

        # Undo
        if key == ord("u"):
            if current_poly:
                removed = current_poly.pop()
                print(f"[INFO] Undo point {removed} (current len={len(current_poly)})")
                redraw()
            elif polylines:
                removed_poly = polylines.pop()
                print(f"[INFO] Undo last polyline with {len(removed_poly)} points. "
                      f"Remaining polylines: {len(polylines)}")
                redraw()
                save_outputs()
            else:
                print("[INFO] Nothing to undo.")

        # 手動保存
        if key == ord("s"):
            print("[INFO] Saving outputs (manual trigger)...")
            save_outputs()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
