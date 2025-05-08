import os
import time
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

ROOT_DIR = "ethz-cil"
CACHE_DIR = os.path.join(ROOT_DIR, "vggt_preds_cache")
EXPERIMENT_ROOT = os.path.join(ROOT_DIR, "experiments")
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

TRAIN_LIST_PATH = os.path.join(ROOT_DIR, "train_list.txt")
TEST_LIST_PATH = os.path.join(ROOT_DIR, "test_list.txt")
TRAIN_DIR = os.path.join(ROOT_DIR, "train/train")
TEST_DIR = os.path.join(ROOT_DIR, "test/test")

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
        CACHE_DIR, split, os.path.basename(image_path).replace("rgb.png", "depth.npy")
    )
    if not os.path.exists(depth_file):
        h, w = img.shape[:2]
        images = load_and_preprocess_images([image_path]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            pred = model(images)["depth"][0, 0, :, :, 0].cpu().numpy()
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        # save the prediction
        os.makedirs(os.path.join(CACHE_DIR, split), exist_ok=True)
        np.save(os.path.join(CACHE_DIR, split, os.path.basename(image_path)), pred)
    else:
        pred_depth = np.load(depth_file)
        pred = torch.from_numpy(pred_depth).unsqueeze(0)

    # print("pred shape", pred.shape)
    # print("img shape", img.shape)
    return pred, img


from skimage.restoration import denoise_tv_chambolle


def anisotropic_diffusion(depth, img):
    return denoise_tv_chambolle(depth.astype(np.float32), weight=0.1)


def joint_bilateral(depth, img):
    depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
    img_np = img.astype(np.float32)

    return cv2.ximgproc.jointBilateralFilter(
        img_np,  # joint/guide image
        depth_np,  # src image to be filtered
        9,  # diameter of each pixel neighborhood
        75,  # sigmaColor
        75,  # sigmaSpace
    )


def run_experiment(experiment_name, refine_func):
    print(f"\n▶ Running experiment: {experiment_name}")
    exp_dir = os.path.join(EXPERIMENT_ROOT, f"vggt_depth_{experiment_name}")
    pred_dir = os.path.join(exp_dir, "test_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "metrics.json")

    train_pairs = load_rgb_depth_pairs(TRAIN_LIST_PATH)
    test_pairs = load_rgb_depth_pairs(TEST_LIST_PATH)

    total_mse, results = 0, []
    t0 = time.time()
    pbar = tqdm(train_pairs, desc=f"Train Eval [{experiment_name}]")
    for idx, (rgb_file, depth_file) in enumerate(pbar):
        # for rgb_file, depth_file in tqdm(train_pairs, desc=f"Train Eval [{experiment_name}], MSE: {total_mse}"):
        rgb_path = os.path.join(TRAIN_DIR, rgb_file)
        gt_depth = np.load(os.path.join(TRAIN_DIR, depth_file))
        raw_depth, img = predict_vggt_depth(rgb_path, "train")
        refined = refine_func(raw_depth, img)
        mse = scale_invariant_mse(refined, gt_depth)
        total_mse += mse
        results.append({"file": rgb_file, "mse": mse})
        avg_mse_so_far = total_mse / (idx + 1)
        pbar.set_postfix(mse_so_far=avg_mse_so_far)

    avg_mse = total_mse / len(train_pairs)
    runtime = time.time() - t0

    with open(log_path, "w") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "avg_si_mse": avg_mse,
                "samples": len(train_pairs),
                "runtime_sec": runtime,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(
        f"{experiment_name} - Final Train SI-MSE: {avg_mse:.4f} | Time: {runtime:.1f}s"
    )

    for rgb_file, depth_file in tqdm(test_pairs, desc=f"Test Save [{experiment_name}]"):
        rgb_path = os.path.join(TEST_DIR, rgb_file)
        raw_depth, img = predict_vggt_depth(rgb_path, "test")
        refined = refine_func(raw_depth, img)
        pred_path = os.path.join(pred_dir, depth_file)
        np.save(pred_path, refined)


if __name__ == "__main__":
    for name, refine_fn in EXPERIMENTS.items():
        try:
            run_experiment(name, refine_fn)
        except Exception as e:
            print(f"Skipped {name} due to error: {e}")
