import os
import time
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
from skimage.restoration import denoise_tv_chambolle
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from depth_refinement_dual_unet import RGBDepthPairs


ROOT_DIR = "../data"
CACHE_DIR = os.path.join(ROOT_DIR, "cached_vggt_raw_depths_and_conf")
EXPERIMENT_ROOT = os.path.join(ROOT_DIR, "experiments")
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

TRAIN_LIST_PATH = os.path.join(ROOT_DIR, "train_list.txt")
TEST_LIST_PATH = os.path.join(ROOT_DIR, "test_list.txt")
TRAIN_DIR = os.path.join(ROOT_DIR, "train/train")
TEST_DIR = os.path.join(ROOT_DIR, "test/test")
TEST_DIR = os.path.join(ROOT_DIR, "enhancement")
# TEST_DIR = os.path.join("lowlightres")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"device: {device}, dtype: {dtype}")

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

EXPERIMENTS = {
    "raw": lambda depth, img: depth,
    "bilateral": lambda depth, img: cv2.bilateralFilter(
         depth.astype(np.float32), 9, 75, 75
    ),
    "anisotropic": lambda depth, img: anisotropic_diffusion(depth, img),
    "joint_bilateral": lambda depth, img: joint_bilateral(depth, img),
}

def load_rgb_depth_pairs(list_path):
    with open(list_path, "r") as f:
        return [line.strip().split() for line in f if line.strip()]


def scale_invariant_mse(pred, gt):
    pred_resized = cv2.resize(
        pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    pred_log = np.log(pred_resized + 1e-6).astype(np.float32)
    gt_log = np.log(gt + 1e-6).astype(np.float32)
    alpha = np.mean(gt_log - pred_log)
    diff = pred_log - gt_log
    return np.sqrt(np.mean((diff + alpha) ** 2))


def predict_vggt_depth(image_path, split):
    img = cv2.imread(image_path)
    depth_file = os.path.join(
        CACHE_DIR, split, "cache_coarse",os.path.basename(image_path).replace(".png", ".depth.npy")
    )
    if not os.path.exists(depth_file):
    # if True:
        h, w = img.shape[:2]
        images = load_and_preprocess_images([image_path]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            pred = model(images)["depth"][0, 0, :, :, 0].cpu().numpy()
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        # save the prediction
        os.makedirs(os.path.join(CACHE_DIR, split, "cache_coarse"), exist_ok=True)
        np.save(os.path.join(CACHE_DIR, split, "cache_coarse", os.path.basename(image_path)), pred)
    else:
        pred_depth = np.load(depth_file)
        pred = torch.from_numpy(pred_depth).unsqueeze(0)

    print("pred shape", pred.shape)
    print("img shape", img.shape)
    return pred, img


def anisotropic_diffusion(depth, img):
    return denoise_tv_chambolle(depth.astype(np.float32), weight=0.1)


def joint_bilateral(depth, img):
    # depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
    depth_np = depth.astype(np.float32)
    img_np = img.astype(np.float32)

    return cv2.ximgproc.jointBilateralFilter(
        img_np,  # joint/guide image
        depth_np,  # src image to be filtered
        9,  # diameter of each pixel neighborhood
        75,  # sigmaColor
        75,  # sigmaSpace
    )

import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
def save_colored_depth(depth, out_path, cmap='plasma'):
    # Load depth map
    depth = depth

    # Normalize to [0, 1]
    print("depth min, max", depth.min(), depth.max())
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # Convert to 8-bit grayscale [0, 255]
    depth_8bit = (depth_norm * 255).astype(np.uint8)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    depth_colored = colormap(depth_8bit)[:, :, :3]  # Drop alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Save as PNG
    cv2.imwrite(out_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))

# Example usage
def run_experiment(experiment_name, refine_func, loader, split_name="train"):
    torch.manual_seed(1881)
    print(f"\n▶ Running experiment: {experiment_name} [{split_name}]")
    exp_dir = os.path.join(EXPERIMENT_ROOT, f"vggt_depth_{experiment_name}")
    pred_dir = os.path.join(exp_dir, f"{split_name}_predictions")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, f"metrics_{split_name}.json")

    total_mse, results = 0, []
    total_mse_normal = 0
    total_si, total_mae, total_d1, total_d2 = 0, 0, 0, 0
    t0 = time.time()
    pbar = tqdm(loader, desc=f"{split_name.capitalize()} Eval [{experiment_name}]")

    for idx, (rgb, gt_depth, rgb_path, d0, c0) in enumerate(pbar):
        rgb_path = rgb_path[0]  # string
        img = (rgb[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_depth = gt_depth[0][0].numpy()
        raw_depth = d0[0][0].numpy()

        refined = refine_func(raw_depth, img)

        out_name = os.path.basename(rgb_path).replace(".png", ".npy")
        pred_path = os.path.join(pred_dir, out_name)
        png_path = pred_path.replace(".npy", ".png")
        #save_colored_depth(refined, png_path, cmap='plasma')

        mse = scale_invariant_mse(refined, gt_depth)
        total_mse += mse
        results.append({"file": os.path.basename(rgb_path), "mse": mse})
        avg_mse_so_far = total_mse / (idx + 1)
        pbar.set_postfix(mse_so_far=avg_mse_so_far)

        eps = 1e-6
        pred_tensor = torch.from_numpy(refined).to(device).float()
        gt_tensor = torch.from_numpy(gt_depth).to(device).float()
        med_pred = torch.median(pred_tensor.view(-1)).clamp(min=eps)
        med_gt = torch.median(gt_tensor.view(-1)).clamp(min=eps)
        scale = med_gt / med_pred
        pred_tensor_scaled = pred_tensor * scale

        silog = torch.mean((torch.log(pred_tensor + eps) - torch.log(gt_tensor + eps)) ** 2) - (
            torch.mean(torch.log(pred_tensor + eps) - torch.log(gt_tensor + eps))
        ) ** 2

        mse_normal = torch.mean((pred_tensor - gt_tensor) ** 2).item()
        mae = torch.mean(torch.abs(pred_tensor_scaled - gt_tensor)).item()
        ratio = torch.max(pred_tensor_scaled / (gt_tensor + eps), gt_tensor / (pred_tensor_scaled + eps))
        delta1 = torch.mean((ratio < 1.25).float()).item()
        delta2 = torch.mean((ratio < 1.25 ** 2).float()).item()

        total_si += silog.item()
        total_mae += mae
        total_mse_normal += mse_normal
        total_d1 += delta1
        total_d2 += delta2

    avg_mse = total_mse / len(loader)
    num_samples = len(loader)
    runtime = time.time() - t0    

    with open(log_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "split": split_name,
            "avg_si_mse": avg_mse,
            "avg_silog": total_si / num_samples,
            "avg_mse_normal": total_mse_normal / num_samples,
            "avg_mae": total_mae / num_samples,
            "avg_delta1": total_d1 / num_samples,
            "avg_delta2": total_d2 / num_samples,
            "samples": num_samples,
            "runtime_sec": runtime,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"{experiment_name} - Final {split_name.capitalize()} SI-MSE: {avg_mse:.4f} | Time: {runtime:.1f}s")

if __name__ == "__main__":
    torch.manual_seed(1881)
    train_ds_full = RGBDepthPairs(root="../data/train/train", list_txt="../data/train_list.txt")
    test_ds = RGBDepthPairs(root="../data/test/test", list_txt="../data/test_list.txt")

    eval_size = int(0.2 * len(train_ds_full))
    print("eval size", eval_size)
    train_ds, eval_ds = random_split(train_ds_full, [len(train_ds_full) - eval_size, eval_size])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    for name, refine_fn in EXPERIMENTS.items():
        try:
            run_experiment(name, refine_fn, train_loader, split_name="train")
            run_experiment(name, refine_fn, eval_loader, split_name="eval")
            # run_experiment(name, refine_fn, test_loader, split_name="test")
        except Exception as e:
            print(f"Skipped {name} due to error: {e}")