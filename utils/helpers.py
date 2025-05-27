import os
import torch
import yaml
from types import SimpleNamespace
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# This function converts a dictionary to a SimpleNamespace object recursively.
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

# Load the configuration file and convert it to a SimpleNamespace object.
def load_config(path="configs/config.yaml"):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)


config = load_config()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


####
# Data augmentation and preprocessing
####

def target_transform(depth):
    """
    Resizes a depth map tensor to [1, H, W] using bilinear interpolation.
    Assumes input is [1, H, W] or [H, W].
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)  # → [1, H, W]
    elif depth.ndim != 3:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    depth = depth.unsqueeze(0)  # → [1, 1, H, W]
    depth = torch.nn.functional.interpolate(
        depth,
        size=config.data.input_size,
        mode='bilinear',
        align_corners=True
    )
    depth = depth.squeeze(0)  # → [1, H, W]
    return depth


def center_crop(enc_feat, target_feat):
    _, _, h, w = enc_feat.shape
    _, _, th, tw = target_feat.shape

    crop_top = max((h - th) // 2, 0)
    crop_left = max((w - tw) // 2, 0)

    return enc_feat[:, :, crop_top:crop_top+th, crop_left:crop_left+tw]


class RandomFlipAndRotation:
    def __call__(self, image):
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
        return image

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

class PairedTransform:
    def __call__(self, image, depth):
        # Convert to tensor only if needed
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)

        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float().unsqueeze(0)
        elif isinstance(depth, Image.Image):
            depth = torch.from_numpy(np.array(depth)).float().unsqueeze(0)
        elif isinstance(depth, torch.Tensor):
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)  # Ensure shape is [1, H, W]
        else:
            raise TypeError(f"Unsupported depth type: {type(depth)}")

        # Apply paired transforms
        if random.random() > 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            depth = TF.rotate(depth, angle)

        return image, depth



# Enable dropout layers during inference for Monte Carlo Dropout
def enable_dropout(model):
    """Enable dropout layers during inference (MC Dropout)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()

# Performs Monte Carlo Dropout for uncertainty estimation
def predict_mc_dropout(model, input_tensor, num_samples=20):
    model.eval()
    enable_dropout(model)  # Enable dropout layers
    preds = []

    deep_supervision = config.model.deep_supervision
    with torch.no_grad():
        for _ in range(num_samples):
            if deep_supervision:
                # Forward pass with deep supervision
                out, _ = model(input_tensor)
            else:
                # Standard forward pass
                out = model(input_tensor)  # shape: [B, 2, H, W]
            mean = out[:, 0:1]
            preds.append(mean)

    preds = torch.stack(preds, dim=0)  # [T, B, 1, H, W]
    mean_pred = preds.mean(dim=0)
    epistemic_uncertainty = preds.var(dim=0)

    return mean_pred, epistemic_uncertainty


####
# Loss functions
####
def aleatoric_loss(pred, target):
    pred_mean = pred[:, 0:1, :, :]
    pred_log_var = pred[:, 1:2, :, :]
    precision = torch.exp(-pred_log_var)

    #regularize the log variance
    lamda = 1e-2
    reg_loss = lamda * torch.mean(pred_log_var ** 2)
    return torch.mean(0.5 * (precision * (pred_mean - target) ** 2 + pred_log_var)) + reg_loss

def laplacian_aleatoric_loss(pred, target):
    """Computes Laplacian negative log-likelihood loss."""
    pred_mean = pred[:, 0:1, :, :]
    log_sigma = pred[:, 1:2, :, :]

    # Clamp log_sigma to avoid exploding/vanishing
    log_sigma = torch.clamp(log_sigma, min=-3.0, max=3.0)

    error = torch.abs(target - pred_mean)
    loss = error / torch.exp(log_sigma) + log_sigma
    mean_loss = loss.mean()

    # --- Runtime logging every N steps ---
    if torch.isnan(mean_loss) or torch.isinf(mean_loss):
        print("⚠️ NaN or Inf detected in loss!")
        print(f"mean(log_sigma): {log_sigma.mean().item():.4f}, std: {log_sigma.std().item():.4f}, "
              f"min: {log_sigma.min().item():.4f}, max: {log_sigma.max().item():.4f}")
        print(f"mean(error): {error.mean().item():.4f}, max(error): {error.max().item():.4f}")
        print(f"mean(pred_mean): {pred_mean.mean().item():.4f}, min: {pred_mean.min().item():.4f}, max: {pred_mean.max().item():.4f}")
        raise ValueError("NaN detected in loss — check log_sigma and predictions.")

    return mean_loss

def scale_invariant_laplacian_aleatoric_loss(pred, target, clamp_log_sigma=True):
    """
    pred: tensor of shape [B, 2, H, W] where:
        pred[:, 0:1, ...] = predicted depth
        pred[:, 1:2, ...] = predicted log_sigma in log-depth space
    target: ground truth depth, shape [B, 1, H, W]
    """
    eps = 1e-6
    pred_mean = pred[:, 0:1, :, :]
    log_sigma = pred[:, 1:2, :, :]

    if clamp_log_sigma:
        log_sigma = torch.clamp(log_sigma, min=-3.0, max=3.0)

    # Log-depth space
    log_pred = torch.log(pred_mean + eps)
    log_target = torch.log(target + eps)

    d = log_pred - log_target
    mu = torch.mean(d, dim=[1, 2, 3], keepdim=True)  # per-image mean
    d_centered = d - mu

    sigma = torch.exp(log_sigma)
    loss = torch.abs(d_centered) / sigma + log_sigma

    return loss.mean()



def compute_weight_norm(model):
    """Compute the squared L2 norm of all model parameters (||θ||²)."""
    norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            norm += torch.sum(param ** 2)
    return norm

def compute_weight_decay(model, dropout_p, batch_size):
    """Compute the weight decay term for the model."""
    weight_norm = compute_weight_norm(model)
    return ((1 - dropout_p) / (2 * batch_size)) * weight_norm

def deep_supervision_loss(main_output, aux_outputs, target, criterion, weights=None):
    """
    main_output: Tensor [B, 2, H, W]
    aux_outputs: List of Tensors [B, 2, H, W]
    target: Ground truth [B, 1, H, W]
    criterion: loss function (e.g., Laplacian heteroscedastic loss)
    weights: Optional list of weights for each aux output
    """
    loss = criterion(main_output, target)
    if weights is None:
        weights = [1.0 / len(aux_outputs)] * len(aux_outputs)
    
    for aux_out, w in zip(aux_outputs, weights):
        loss += w * criterion(aux_out, target)
    
    return loss


# to make depth map less patchy
def compute_smoothness_loss(predicted_depth, input_image):
    """
    predicted_depth: [B, 1, H, W]
    input_image: [B, 3, H, W] (RGB)

    Returns:
        Edge-aware smoothness loss.
    """
    depth_grad_x = torch.abs(predicted_depth[:, :, :, :-1] - predicted_depth[:, :, :, 1:])
    depth_grad_y = torch.abs(predicted_depth[:, :, :-1, :] - predicted_depth[:, :, 1:, :])

    image_grad_x = torch.mean(torch.abs(input_image[:, :, :, :-1] - input_image[:, :, :, 1:]), 1, keepdim=True)
    image_grad_y = torch.mean(torch.abs(input_image[:, :, :-1, :] - input_image[:, :, 1:, :]), 1, keepdim=True)

    weights_x = torch.exp(-image_grad_x)
    weights_y = torch.exp(-image_grad_y)

    smoothness_x = depth_grad_x * weights_x
    smoothness_y = depth_grad_y * weights_y

    return smoothness_x.mean() + smoothness_y.mean()


# Visualisation
def normalize_for_vis(tensor, vmin=0.0, vmax=10.0):
    return torch.clamp((tensor - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

def visualize_prediction(rgb, depth, save_path, uncertainties=None):
    plt.figure(figsize=(12, 4))
    ncols = 1 + (1 if rgb is not None else 0) + (len(uncertainties) if uncertainties else 0)

    idx = 1
    if rgb is not None:
        plt.subplot(1, ncols, idx)

        # ✅ De-normalize (adjust if needed!)
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device)[:, None, None]
        rgb_denorm = (rgb * std + mean).clamp(0, 1)

        rgb_np = TF.to_pil_image(rgb_denorm.cpu())
        plt.imshow(rgb_np)
        plt.title("RGB Input")
        plt.axis('off')
        idx += 1

    plt.subplot(1, ncols, idx)
    plt.imshow(depth, cmap='inferno')
    plt.title("Predicted Depth")
    plt.axis('off')
    idx += 1

    if uncertainties:
        for key, value in uncertainties.items():
            plt.subplot(1, ncols, idx)
            plt.imshow(value, cmap='magma')
            plt.title(f"{key.capitalize()} Uncertainty")
            plt.axis('off')
            idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# postprocess depth map
def postprocess_depth(depth: np.ndarray, rgb: np.ndarray = None, uncertainty: np.ndarray = None) -> np.ndarray:
    """Applies uncertainty-guided fusion, guided filtering, and denoising to depth map."""
    
    # --- 1. Clamp depth values ---
    depth = np.clip(depth, 0.01, 10.0)

    # --- 2. Uncertainty-guided fusion (optional) ---
    if uncertainty is not None:
        weight = np.exp(-uncertainty)
        depth = weight * depth + (1 - weight) * cv2.medianBlur(depth, 5)  # fallback: smooth prior

    # --- 3. Median filter ---
    depth = cv2.medianBlur(depth.astype(np.float32), ksize=5)

    # --- 4. Bilateral filter ---
    depth = cv2.bilateralFilter(depth, d=9, sigmaColor=75, sigmaSpace=75)

    # --- 5. Guided filter (if RGB is available) ---
    if rgb is not None:
        try:
            import cv2.ximgproc as xip
            rgb_uint8 = np.clip((rgb * 255).astype(np.uint8), 0, 255)  # Convert to 0–255
            if rgb_uint8.shape[0] == 3:  # CHW -> HWC
                rgb_uint8 = np.transpose(rgb_uint8, (1, 2, 0))

            depth_guide = xip.guidedFilter(guide=rgb_uint8, src=depth.astype(np.float32),
                                           radius=9, eps=1e-2)
            depth = depth_guide
        except ImportError:
            print("Guided filter not available (cv2.ximgproc). Skipping...")

    return depth



# not used
def training_step_with_mc(model, x, y, T=5, criterion=aleatoric_loss):
    model.train()
    loss = 0.0
    for _ in range(T):
        out = model(x)  # stochastic forward
        loss += criterion(out, y)
    return loss / T