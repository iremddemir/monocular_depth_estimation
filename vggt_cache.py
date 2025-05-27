import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

ROOT_DIR = "data"
CACHE_DIR = os.path.join(ROOT_DIR, "vggt_preds_cache")
os.makedirs(os.path.join(CACHE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "test"), exist_ok=True)

TRAIN_LIST_PATH = os.path.join(ROOT_DIR, "train_list.txt")
TEST_LIST_PATH = os.path.join(ROOT_DIR, "test_list.txt")
TRAIN_DIR = os.path.join(ROOT_DIR, "train/train")
TEST_DIR = os.path.join(ROOT_DIR, "test/test")

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()


def load_pairs(list_path):
    with open(list_path, "r") as f:
        return [line.strip().split() for line in f if line.strip()]


def save_predictions(pairs, split):
    for rgb_file, depth_file in tqdm(pairs, desc=f"VGGT {split} preds"):
        out_path = os.path.join(CACHE_DIR, split, depth_file)
        if os.path.exists(out_path):
            continue
        rgb_path = os.path.join(TRAIN_DIR if split == "train" else TEST_DIR, rgb_file)
        img = cv2.imread(rgb_path)
        h, w = img.shape[:2]
        images = load_and_preprocess_images([rgb_path]).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            pred = model(images)["depth"][0, 0, :, :, 0].cpu().numpy()
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        np.save(out_path, pred_resized)


if __name__ == "__main__":
    save_predictions(load_pairs(TRAIN_LIST_PATH), "train")
    save_predictions(load_pairs(TEST_LIST_PATH), "test")
