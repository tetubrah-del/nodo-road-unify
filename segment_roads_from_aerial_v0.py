import os
from pathlib import Path

import numpy as np
from PIL import Image

# ==== 設定 ====
IMAGE_PATH = "/mnt/data/aerial_tsujiwara_10km_approx.jpg"
OUTPUT_MASK_PATH = "road_mask_tsujiwara.png"

TILE_SIZE = 512      # タイルの一辺（ピクセル）
TILE_OVERLAP = 64    # オーバーラップ（境目をなめらかにする用）

# ==== ここを後でAIモデル実装に差し替える ====
def segment_tile(tile_rgb: np.ndarray) -> np.ndarray:
    """
    1タイル分のRGB画像 (H, W, 3, uint8) を受け取り、
    同サイズのバイナリマスク (H, W, uint8) を返す。
    値：0 = 非農道, 255 = 農道

    今はダミー実装（全て0）にしておく。
    後で SAM / U-Net などに差し替える。
    """
    h, w, _ = tile_rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # TODO:
    # - SAM2 を使う場合：ここで model.predict(tile_rgb) して roadっぽい領域を抽出
    # - U-Net を使う場合：前処理 → model() → 後処理 で 0/255 マスクに変換
    return mask


def tile_image(image: np.ndarray, tile_size: int, overlap: int):
    """
    大きな画像をタイルに分割して yield するジェネレータ。
    """
    h, w, _ = image.shape
    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y+tile_size, x:x+tile_size, :]
            # 端っこでサイズが小さくなったタイルもそのまま返す
            yield (x, y), tile


def merge_tiles_to_mask(
    image_shape, tile_size, overlap, tile_masks
):
    """
    タイルごとのマスクを元の画像サイズに再合成する。
    tile_masks: list of ((x, y), mask_array)
    """
    h, w, _ = image_shape
    full_mask = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    step = tile_size - overlap

    for (x, y), mask in tile_masks:
        mh, mw = mask.shape
        full_mask[y:y+mh, x:x+mw] += mask.astype(np.float32)
        weight[y:y+mh, x:x+mw] += 1.0

    # 平均をとって 0/255 に戻す
    weight[weight == 0] = 1.0
    full_mask = full_mask / weight
    full_mask = (full_mask > 127).astype(np.uint8) * 255

    return full_mask


def main():
    # 1. 航空写真読み込み
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(img)
    print("Input image shape:", img_np.shape)

    # 2. タイルに分割して順次セグメンテーション
    tile_masks = []
    for (x, y), tile in tile_image(img_np, TILE_SIZE, TILE_OVERLAP):
        mask = segment_tile(tile)
        tile_masks.append(((x, y), mask))

    # 3. タイルマスクを1枚のマスクに再合成
    full_mask = merge_tiles_to_mask(img_np.shape, TILE_SIZE, TILE_OVERLAP, tile_masks)

    # 4. PNGとして保存
    mask_img = Image.fromarray(full_mask, mode="L")
    mask_img.save(OUTPUT_MASK_PATH)
    print("Saved road mask to:", OUTPUT_MASK_PATH)


if __name__ == "__main__":
    main()
