#!/usr/bin/env python
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor


def load_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load image and return (bgr, rgb, hsv)."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return bgr, rgb, hsv


def make_hsv_road_mask(
    hsv: np.ndarray,
    sat_max: int = 60,
    val_min: int = 120,
    val_max: int = 255,
) -> np.ndarray:
    """
    Create a coarse mask of "road-like" pixels in HSV space.

    - 低彩度 (S が小さい) → グレー系・アスファルトっぽい
    - 高輝度 (V が高い) → 影や黒い部分を除外
    """
    h, s, v = cv2.split(hsv)

    road_mask = (s <= sat_max) & (v >= val_min) & (v <= val_max)
    road_mask = road_mask.astype(np.uint8) * 255

    # 軽くクロージングで穴埋め
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return road_mask


def sample_seed_points(
    mask: np.ndarray,
    grid_size: int = 32,
    max_points: int = 200,
) -> List[Tuple[int, int]]:
    """
    Self-Prompt 用に seed となる点をマスク上からサンプルする。

    - 画像を grid_size ピクセル単位でグリッド分割
    - 各グリッド内で 1 点だけ代表点を取る（道路画素がある場合のみ）
    - 全体で max_points を超えないように間引き
    """
    h, w = mask.shape
    points: List[Tuple[int, int]] = []

    for gy in range(0, h, grid_size):
        for gx in range(0, w, grid_size):
            sub = mask[gy : gy + grid_size, gx : gx + grid_size]
            ys, xs = np.where(sub > 0)
            if len(xs) == 0:
                continue
            # グリッド内の中央付近の点を 1 つ選ぶ
            idx = len(xs) // 2
            px = int(gx + xs[idx])
            py = int(gy + ys[idx])
            points.append((px, py))

    # 多すぎる場合は間引き
    if len(points) > max_points:
        step = len(points) / max_points
        points = [points[int(i * step)] for i in range(max_points)]

    return points


def run_sam_from_points(
    predictor: SamPredictor,
    rgb_image: np.ndarray,
    points: List[Tuple[int, int]],
    pred_iou_thresh: float = 0.5,
) -> np.ndarray:
    """
    SAM にシード点を順番に投げて、すべてのマスクを OR 結合して返す。
    """
    if len(points) == 0:
        raise ValueError("No seed points were found for SAM. "
                         "Try relaxing HSV thresholds or grid_size.")

    predictor.set_image(rgb_image)

    h, w, _ = rgb_image.shape
    accumulated_mask = np.zeros((h, w), dtype=np.uint8)

    for (x, y) in points:
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # 一番 IoU スコアが高いマスクだけ使う
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < pred_iou_thresh:
            continue

        mask = masks[best_idx].astype(np.uint8)
        accumulated_mask = np.maximum(accumulated_mask, mask)

    accumulated_mask = (accumulated_mask > 0).astype(np.uint8) * 255

    return accumulated_mask


def filter_long_thin_components(
    mask: np.ndarray,
    min_area: int = 50,
    max_area: int = 50000,
    min_aspect_ratio: float = 2.0,
) -> np.ndarray:
    """
    ラベリングして「細長い」コンポーネントだけ残す。
    - 面積が小さすぎる／大きすぎるものは除外
    - バウンディングボックスのアスペクト比が min_aspect_ratio 以上のものだけ残す
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)

    for label in range(1, num_labels):  # 0 は背景
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))  # 0 division guard
        aspect_ratio = long_side / short_side

        if aspect_ratio < min_aspect_ratio:
            continue

        out[labels == label] = 255

    return out


def make_overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    元画像に道路マスクをオーバーレイする。
    mask > 0 の部分を赤っぽく塗る。
    """
    color = np.zeros_like(bgr)
    # BGR の R チャンネルを強く
    color[:, :, 2] = 255

    mask_bool = mask > 0
    overlay = bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(
        bgr[mask_bool], 1 - alpha, color[mask_bool], alpha, 0
    )
    return overlay


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic road segmentation from Google satellite image "
                    "using SAM + HSV + shape filters."
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to input satellite image (e.g., data/tsuji_google.png)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to SAM model checkpoint (e.g., data/sam_vit_b.pth)",
    )
    parser.add_argument(
        "--model-type",
        default="vit_b",
        help="SAM model type (e.g., vit_b, vit_l, vit_h). Default: vit_b",
    )
    parser.add_argument(
        "--output-mask",
        required=True,
        help="Path to output binary road mask PNG (e.g., sam_roads_mask.png)",
    )
    parser.add_argument(
        "--output-overlay",
        required=True,
        help="Path to output overlay PNG (e.g., sam_roads_overlay.png)",
    )

    # HSV スカラー
    parser.add_argument("--sat-max", type=int, default=60,
                        help="Max saturation for road HSV mask (default: 60)")
    parser.add_argument("--val-min", type=int, default=120,
                        help="Min value/brightness for road HSV mask (default: 120)")
    parser.add_argument("--val-max", type=int, default=255,
                        help="Max value/brightness for road HSV mask (default: 255)")

    # Self-Prompt サンプリング
    parser.add_argument("--grid-size", type=int, default=32,
                        help="Grid size (px) for seed sampling (default: 32)")
    parser.add_argument("--max-points", type=int, default=200,
                        help="Maximum number of SAM seed points (default: 200)")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.5,
                        help="Minimum IoU score for accepting SAM mask (default: 0.5)")

    # 形状フィルタ
    parser.add_argument("--min-area", type=int, default=50,
                        help="Minimum connected-component area to keep (default: 50)")
    parser.add_argument("--max-area", type=int, default=50000,
                        help="Maximum connected-component area to keep (default: 50000)")
    parser.add_argument("--min-aspect-ratio", type=float, default=2.0,
                        help="Minimum aspect ratio (long/short) to keep (default: 2.0)")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"[INFO] Loading image: {args.image_path}")
    bgr, rgb, hsv = load_image(args.image_path)

    print("[INFO] Building HSV road-likeness mask...")
    hsv_mask = make_hsv_road_mask(
        hsv,
        sat_max=args.sat_max,
        val_min=args.val_min,
        val_max=args.val_max,
    )

    print("[INFO] Sampling seed points for SAM (Self-Prompt)...")
    seed_points = sample_seed_points(
        hsv_mask,
        grid_size=args.grid_size,
        max_points=args.max_points,
    )
    print(f"[INFO] Number of seed points: {len(seed_points)}")

    print("[INFO] Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    predictor = SamPredictor(sam)

    print("[INFO] Running SAM for each seed point...")
    sam_mask = run_sam_from_points(
        predictor,
        rgb_image=rgb,
        points=seed_points,
        pred_iou_thresh=args.pred_iou_thresh,
    )

    print("[INFO] Refining mask with HSV road mask...")
    refined_mask = cv2.bitwise_and(sam_mask, hsv_mask)

    print("[INFO] Filtering long-thin components (road-like shapes)...")
    road_mask = filter_long_thin_components(
        refined_mask,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect_ratio=args.min_aspect_ratio,
    )

    print(f"[INFO] Saving road mask to: {args.output_mask}")
    cv2.imwrite(args.output_mask, road_mask)

    print("[INFO] Creating overlay...")
    overlay = make_overlay(bgr, road_mask, alpha=0.6)
    print(f"[INFO] Saving overlay to: {args.output_overlay}")
    cv2.imwrite(args.output_overlay, overlay)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
