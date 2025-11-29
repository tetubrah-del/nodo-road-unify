#!/usr/bin/env python
import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor


# ------------------
# 画像読み込み
# ------------------

def load_image(image_path: str):
    """Load image and return (bgr, rgb, hsv, gray)."""
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, rgb, hsv, gray


# ------------------
# HSV マスク
# ------------------

def make_hsv_road_mask(
    hsv: np.ndarray,
    sat_min: int = 20,
    sat_max: int = 180,
    val_min: int = 40,
    val_max: int = 220,
) -> np.ndarray:
    """
    「道路っぽい色」をゆるめに拾う HSV マスク。
    辻原の農道は土色なので S, V の範囲を広めにとる。
    """
    h, s, v = cv2.split(hsv)
    road_mask = (
        (s >= sat_min) & (s <= sat_max) &
        (v >= val_min) & (v <= val_max)
    )
    road_mask = road_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return road_mask


# ------------------
# Edge-based Self-Prompt
# ------------------

def sample_seed_points_edge_based(
    gray: np.ndarray,
    hsv_mask: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iter: int = 2,
    min_edge_area: int = 30,
    max_edge_area: int = 50000,
    min_edge_aspect: float = 2.0,
    max_points: int = 400,
    max_points_per_comp: int = 3,
) -> List[Tuple[int, int]]:
    """
    Canny エッジ + 連結成分解析から Self-Prompt 用 seed 点を生成する。
    - HSV マスクで「色的にありえない領域」を先に除外
    - 細長いコンポーネントだけを線候補とみなし、そのエッジ上から数点サンプル
    """
    # Canny エッジ
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    # HSV マスクでフィルタ（農道と無関係な領域を削る）
    edges = cv2.bitwise_and(edges, edges, mask=hsv_mask)

    if np.count_nonzero(edges) == 0:
        return []

    # エッジを膨張して連結を強める
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    h_img, w_img = gray.shape
    points: List[Tuple[int, int]] = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < min_edge_area or area > max_edge_area:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side

        if aspect < min_edge_aspect:
            continue

        # このコンポーネントに属する「元のエッジ」上の座標を取得
        comp_mask = (labels == label).astype(np.uint8)
        comp_edges = cv2.bitwise_and(edges, edges, mask=comp_mask)
        ys, xs = np.where(comp_edges > 0)
        if len(xs) == 0:
            continue

        # コンポーネント内で max_points_per_comp 個まで均等にサンプル
        step = max(1, len(xs) // max_points_per_comp)
        for idx in range(0, len(xs), step):
            px = int(xs[idx])
            py = int(ys[idx])
            # 画像範囲を一応チェック
            if 0 <= px < w_img and 0 <= py < h_img:
                points.append((px, py))
            if len(points) >= max_points:
                break
        if len(points) >= max_points:
            break

    # 重複を少し削る（座標の近さは気にせず単純間引き）
    if len(points) > max_points:
        step = len(points) / max_points
        points = [points[int(i * step)] for i in range(max_points)]

    return points


# ------------------
# SAM 実行
# ------------------

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
        raise ValueError(
            "No seed points were found for SAM. "
            "Try relaxing Canny / HSV / shape thresholds."
        )

    predictor.set_image(rgb_image)

    h, w, _ = rgb_image.shape
    accumulated = np.zeros((h, w), dtype=np.uint8)

    for (x, y) in points:
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        if scores[best_idx] < pred_iou_thresh:
            continue

        mask = masks[best_idx].astype(np.uint8)
        accumulated = np.maximum(accumulated, mask)

    accumulated = (accumulated > 0).astype(np.uint8) * 255
    return accumulated


# ------------------
# 形状フィルタ（細長いコンポーネント）
# ------------------

def filter_long_thin_components(
    mask: np.ndarray,
    min_area: int = 80,
    max_area: int = 200000,
    min_aspect_ratio: float = 2.5,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue

        long_side = max(w, h)
        short_side = max(1, min(w, h))
        aspect = long_side / short_side

        if aspect < min_aspect_ratio:
            continue

        out[labels == label] = 255

    return out


# ------------------
# オーバーレイ
# ------------------

def make_overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    color = np.zeros_like(bgr)
    color[:, :, 2] = 255  # 赤
    mask_bool = mask > 0

    overlay = bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(
        bgr[mask_bool], 1 - alpha, color[mask_bool], alpha, 0
    )
    return overlay


# ------------------
# CLI
# ------------------

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Automatic farm-road segmentation from Google satellite "
                    "using SAM + edge-based self-prompt + HSV + shape filters."
    )
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_b",
                        help="vit_b / vit_l / vit_h (default: vit_b)")
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--output-overlay", required=True)

    # HSV
    parser.add_argument("--sat-min", type=int, default=20)
    parser.add_argument("--sat-max", type=int, default=180)
    parser.add_argument("--val-min", type=int, default=40)
    parser.add_argument("--val-max", type=int, default=220)

    # Canny + エッジ
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--dilate-iter", type=int, default=2)
    parser.add_argument("--min-edge-area", type=int, default=30)
    parser.add_argument("--max-edge-area", type=int, default=50000)
    parser.add_argument("--min-edge-aspect", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument("--max-points-per-comp", type=int, default=3)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.5)

    # 形状フィルタ
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--max-area", type=int, default=200000)
    parser.add_argument("--min-aspect-ratio", type=float, default=2.5)

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"[INFO] Loading image: {args.image_path}")
    bgr, rgb, hsv, gray = load_image(args.image_path)

    print("[INFO] Building HSV coarse road mask...")
    hsv_mask = make_hsv_road_mask(
        hsv,
        sat_min=args.sat_min,
        sat_max=args.sat_max,
        val_min=args.val_min,
        val_max=args.val_max,
    )

    print("[INFO] Sampling seed points (edge-based self-prompt)...")
    seed_points = sample_seed_points_edge_based(
        gray=gray,
        hsv_mask=hsv_mask,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        dilate_iter=args.dilate_iter,
        min_edge_area=args.min_edge_area,
        max_edge_area=args.max_edge_area,
        min_edge_aspect=args.min_edge_aspect,
        max_points=args.max_points,
        max_points_per_comp=args.max_points_per_comp,
    )
    print(f"[INFO] Seed points: {len(seed_points)}")

    print("[INFO] Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    predictor = SamPredictor(sam)

    print("[INFO] Running SAM segmentation...")
    sam_mask = run_sam_from_points(
        predictor=predictor,
        rgb_image=rgb,
        points=seed_points,
        pred_iou_thresh=args.pred_iou_thresh,
    )

    print("[INFO] Refining with HSV mask...")
    refined = cv2.bitwise_and(sam_mask, hsv_mask)

    print("[INFO] Filtering long-thin components (road-like)...")
    road_mask = filter_long_thin_components(
        refined,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect_ratio=args.min_aspect_ratio,
    )

    print(f"[INFO] Saving road mask: {args.output_mask}")
    cv2.imwrite(args.output_mask, road_mask)

    print("[INFO] Creating overlay...")
    overlay = make_overlay(bgr, road_mask, alpha=0.6)
    print(f"[INFO] Saving overlay: {args.output_overlay}")
    cv2.imwrite(args.output_overlay, overlay)

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
