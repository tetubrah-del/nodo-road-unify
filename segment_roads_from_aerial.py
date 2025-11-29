import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# =========================
# 設定
# =========================

# ★ここをあなたの環境のパスに変更してください
IMAGE_PATH = "./data/aerial_tsujiwara_10km_approx.jpg"

OUTPUT_MASK_PATH = "road_mask_tsujiwara.png"

TILE_SIZE = 512      # タイル一辺（ピクセル）
TILE_OVERLAP = 64    # タイルの重なり（境界なめらか用）

# =========================
# DeepLabV3 モデル準備
# =========================

# =========================
# SpaceNet Road U-Net モデル準備
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPACENET_WEIGHTS = "models/spacenet_unet_resnet34.pth"

print("Loading SpaceNet U-Net model on", device)

# U-Net (ResNet34 encoder) — 出力1ch（road=1, background=0）
spacenet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # すでに学習済み重みを読み込む前提
    in_channels=3,
    classes=1
).to(device)

state = torch.load(SPACENET_WEIGHTS, map_location=device)
spacenet_model.load_state_dict(state)
spacenet_model.eval()

# 入力タイルの前処理（ImageNet 正規化）
tile_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])



# =========================
# タイル→マスク（セグメンテーション）
# =========================

def segment_tile(tile_rgb: np.ndarray) -> np.ndarray:
    """
    SpaceNet の道路 U-Net を使って、
    1タイル分のRGB画像 (H, W, 3, uint8) から
    同サイズのバイナリマスク (H, W, uint8, 0/255) を返す。
    """

    h, w, _ = tile_rgb.shape

    # ---- 前処理 ----
    inp = tile_transform(tile_rgb).unsqueeze(0).to(device)  # (1, 3, 512, 512)

    # ---- 推論 ----
    with torch.no_grad():
        logits = spacenet_model(inp)          # (1, 1, 512, 512)
        probs = torch.sigmoid(logits)[0, 0]   # (512, 512)

    prob_map = probs.cpu().numpy()

    # ---- 後処理：元サイズに戻して二値化 ----
    prob_resized = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # しきい値はあとで調整。まずは 0.5 で。
    mask = (prob_resized > 0.5).astype(np.uint8) * 255

    return mask



# =========================
# タイル分割・マージ
# =========================

def tile_image(image: np.ndarray, tile_size: int, overlap: int):
    """
    大きな画像をタイルに分割して yield するジェネレータ。
    """
    h, w, _ = image.shape
    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y + tile_size, x:x + tile_size, :]
            yield (x, y), tile


def merge_tiles_to_mask(image_shape, tile_size, overlap, tile_masks):
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
        full_mask[y:y + mh, x:x + mw] += mask.astype(np.float32)
        weight[y:y + mh, x:x + mw] += 1.0

    # 平均をとって 0/255 に
    weight[weight == 0] = 1.0
    full_mask = full_mask / weight
    full_mask = (full_mask > 127).astype(np.uint8) * 255

    return full_mask.astype(np.uint8)


# =========================
# メイン
# =========================

def main():
    img_path = Path(IMAGE_PATH)
    assert img_path.exists(), f"画像が見つかりません: {img_path}"

    print("Loading image:", img_path)
    img = Image.open(str(img_path)).convert("RGB")
    img_np = np.array(img)
    print("Input image shape:", img_np.shape)

    tile_masks = []
    for (x, y), tile in tile_image(img_np, TILE_SIZE, TILE_OVERLAP):
        print(f"Segmenting tile at x={x}, y={y}, shape={tile.shape}")
        mask = segment_tile(tile)
        tile_masks.append(((x, y), mask))

    full_mask = merge_tiles_to_mask(img_np.shape, TILE_SIZE, TILE_OVERLAP, tile_masks)

    # 保存
    mask_img = Image.fromarray(full_mask, mode="L")
    mask_img.save(OUTPUT_MASK_PATH)
    print("Saved road mask to:", OUTPUT_MASK_PATH)


if __name__ == "__main__":
    main()
