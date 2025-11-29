import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

IMAGE_PATH = "data/aerial_tsujiwara_1km_z17.jpg"
OUTPUT_MASK = "road_mask_deeplab.png"
OUTPUT_SKEL = "road_mask_deeplab_skel.png"

device = "cpu"

# DeepLabV3-ResNet50 (COCO)
model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def segment(img):
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)["out"][0]
    return out.argmax(0).cpu().numpy()

def skeletonize(bin_img):
    size = np.size(bin_img)
    skel = np.zeros(bin_img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    while not done:
        eroded = cv2.erode(bin_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(bin_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        bin_img = eroded.copy()

        zeros = size - cv2.countNonZero(bin_img)
        if zeros == size:
            done = True

    return skel

def main():
    assert Path(IMAGE_PATH).exists(), f"not found: {IMAGE_PATH}"

    img = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(img)

    print("segmenting...")
    mask = segment(img)

    # COCO では道路クラスは 0=背景 の隣でなく "pavement/road-ish" が 7 or 8 付近
    # 実験的に 7, 8 を抽出
    road = np.logical_or(mask == 7, mask == 8).astype(np.uint8) * 255

    cv2.imwrite(OUTPUT_MASK, road)
    print("saved:", OUTPUT_MASK)

    skel = skeletonize(road)
    cv2.imwrite(OUTPUT_SKEL, skel)
    print("saved:", OUTPUT_SKEL)

if __name__ == "__main__":
    main()
