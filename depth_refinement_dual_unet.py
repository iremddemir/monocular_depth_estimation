from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import itertools

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import random_split, DataLoader

import numpy as np

from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers import UNet2DModel
from transformers import CLIPVisionModel, CLIPImageProcessor


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

import os
import random

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for full determinism

    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(1881)


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module

def assert_finite(x, name="tensor"):
    if not torch.isfinite(x).all():
        print(f"WARNING: {name} has NaNs/Infs! min={x.min().item()} max={x.max().item()}")
        raise ValueError(f"{name} is not finite!")

vggt = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda")
def vggt_get_depth(image_path, h, w) -> Tuple[Tensor, Tensor]:

    images = load_and_preprocess_images([image_path]).to(device)
    with torch.no_grad():
        depth = vggt(images)["depth"][0, 0, :, :, 0].cpu().numpy()
        conf = vggt(images)["depth_conf"][0, 0, :, :].cpu().numpy()
    
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

    return torch.tensor(depth), torch.tensor(conf)


class DepthRefinementPipeline(DiffusionPipeline):

    def __init__(
        self,
        *,
        tex_unet: UNet2DConditionModel | None = None,
        sem_unet: UNet2DConditionModel | None = None,
        gate_conv:    nn.Conv2d              | None = None,
        clip_model: CLIPVisionModel | None = None,
        semantic_adapter: nn.Sequential | None = None,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        sem_prior_channels: int = 64,
        unet_channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 256),
        layers_per_block: int = 2,
    ):
        super().__init__()

        self.register_to_config(
            clip_model_name=clip_model_name,
            sem_prior_channels=sem_prior_channels,
            unet_channels=unet_channels,
            layers_per_block=layers_per_block,
        )
        
        # Frozen CLIP
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_model = clip_model or _freeze(CLIPVisionModel.from_pretrained(clip_model_name))
        self.semantic_adapter = semantic_adapter or nn.Sequential(
            nn.Linear(self.clip_model.config.hidden_size, sem_prior_channels),
            nn.SiLU(),
        )

        # for m in self.semantic_adapter.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight, gain=0.1)
        #         nn.init.constant_(m.bias, 0)

        # Dual conditional U‑Nets
        in_ch = 3 + 1 + 1  # RGB + D0 + conf

        # Texture UNet (no attention)
        self.tex_unet = tex_unet or UNet2DModel(
            sample_size=None,
            in_channels=in_ch,
            out_channels=1,
            layers_per_block=layers_per_block,
            block_out_channels=unet_channels,
            norm_num_groups=32,
            down_block_types=["DownBlock2D"] * len(unet_channels),
            up_block_types  =["UpBlock2D"]   * len(unet_channels),
        )

        # Semantic UNet with cross-attention
        self.sem_unet = sem_unet or UNet2DConditionModel(
            sample_size=None,
            in_channels=in_ch,
            out_channels=1,
            layers_per_block=layers_per_block,
            block_out_channels=unet_channels,
            norm_num_groups=32,
            cross_attention_dim=sem_prior_channels,
            down_block_types=[
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
            ],
            up_block_types=[
                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ],
        )


        # def init_weights(m):
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.kaiming_normal_(m.weight, a=0.1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.GroupNorm):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # self.tex_unet.apply(init_weights)
        # self.sem_unet.apply(init_weights)

        # Confidence‑aware fusion
        self.gate_conv = gate_conv or nn.Conv2d(2, 1, 3, padding=1)
        nn.init.constant_(self.gate_conv.bias, 0.0)
        nn.init.constant_(self.gate_conv.weight, 0.0)

        self.register_modules(
            clip_model=self.clip_model,
            semantic_adapter=self.semantic_adapter,
            tex_unet=self.tex_unet,
            sem_unet=self.sem_unet,
            gate_conv=self.gate_conv,
        )


    def forward(self, *, image: Tensor, d0 = None, c0 = None, img_path: str = "") -> Dict[str, Tensor]:
        device, dtype = image.device, image.dtype
        B, _, H0, W0 = image.shape 

        # ------------------------------------------------------------
        # 1. Coarse VGGT prediction
        # ------------------------------------------------------------
        if d0 is None or c0 is None:
            d0, c0 = vggt_get_depth(img_path, H0, W0)      #  (H0, W0) 
            d0 = d0.unsqueeze(0).unsqueeze(0)
            c0 = c0.unsqueeze(0).unsqueeze(0)
        # else:
        #     d0 = d0.squeeze(0).squeeze(0)  # (1,1,H0,W0)
        #     c0 = c0.squeeze(0).squeeze(0)  # (1,1,H0,W0)
        d0, c0 = d0.to(device=device, dtype=dtype), c0.to(device=device, dtype=dtype)
        sigma_c0 = torch.sigmoid(c0)

        # ------------------------------------------------------------
        # 2. Global semantics (CLS token)
        # ------------------------------------------------------------
        clip_in = self.clip_processor(
            images=(image * 255).clamp(0, 255).to(torch.uint8),
            return_tensors="pt"
        ).pixel_values.to(device)
        cls      = self.clip_model(pixel_values=clip_in).last_hidden_state[:, 0]  # (B,1024)
        cls = F.normalize(cls, dim=1)
        sem_vec  = self.semantic_adapter(cls)                                     # (B,64)

        assert torch.isfinite(sem_vec).all(), "NaNs in semantic adapter output"

        # ------------------------------------------------------------
        # 3. Build 5-channel core tensor and pad
        # ------------------------------------------------------------
        x_core = torch.cat((image, d0, sigma_c0), dim=1)  # (B,5,H0,W0)

        # compute padding so spatial dims are divisible by 2**levels
        factor     = 2 ** len(self.config.unet_channels)  # e.g. 64
        H_pad      = math.ceil(H0 / factor) * factor
        W_pad      = math.ceil(W0 / factor) * factor
        pad_bottom = H_pad - H0
        pad_right  = W_pad - W0

        x_core = F.pad(x_core, (0, pad_right, 0, pad_bottom), mode="reflect")   # (B,5,H_pad,W_pad)

        # ------------------------------------------------------------
        # 4. Texture branch (full res)
        # ------------------------------------------------------------
        t0 = torch.zeros((B,), dtype=torch.long, device=device)
        d_tex = self.tex_unet(sample=x_core, timestep=t0).sample                # (B,1,H_pad,W_pad)

        # ------------------------------------------------------------
        # 5. Semantic branch (¼ res, cross-attn)
        # ------------------------------------------------------------
        x_sem_low = F.interpolate(x_core, scale_factor=0.25, mode="bilinear", align_corners=False)
        d_sem_low = self.sem_unet(
            sample=x_sem_low,
            timestep=t0,
            encoder_hidden_states=sem_vec.unsqueeze(1)   # (B,1,64)  seq_len=1
        ).sample                                         # (B,1,H_pad/4,W_pad/4)
        
        assert_finite(d_sem_low, "d_sem_low after UNet")
        
        d_sem = F.interpolate(d_sem_low, size=(H_pad, W_pad), mode="bilinear", align_corners=False)

        def process_residual(res: torch.Tensor, eps=1e-6) -> torch.Tensor:
            # mean = res.mean(dim=[2, 3], keepdim=True)
            # std  = res.std(dim=[2, 3], keepdim=True).clamp(min=eps)
            # return (res - mean) / std
            return res.clamp(-10, 10)

        max_change = d0.abs().max() * 0.05  # 10% of coarse depth range
        d_tex = max_change * torch.tanh(process_residual(d_tex))
        d_sem = max_change * torch.tanh(process_residual(d_sem))

        # ------------------------------------------------------------
        # 6. Confidence-aware fusion
        # ------------------------------------------------------------
        # alpha = torch.sigmoid(self.gate_conv(torch.cat((d_tex, d_sem), dim=1)))   # (B,1,H_pad,W_pad)
        gate_in = torch.cat((d_tex, d_sem), dim=1)
        gate_out = self.gate_conv(gate_in)
        gate_out = torch.clamp(gate_out, min=-20, max=20)  # sigmoid is numerically stable in this range
        alpha = torch.sigmoid(gate_out)

        # alpha = torch.ones_like(alpha)  # for d_tex
        # alpha = torch.zeros_like(alpha) # for d_sem
        
        d_tex_crop  = d_tex [..., :H0, :W0]
        d_sem_crop  = d_sem [..., :H0, :W0]
        alpha_crop  = alpha [..., :H0, :W0]
        d_hat_pad = d0 + alpha_crop * d_tex_crop + (1 - alpha_crop) * d_sem_crop         # (B,1,H_pad,W_pad)

        # ------------------------------------------------------------
        # 7. Remove padding, return native resolution
        # ------------------------------------------------------------
        d_hat = d_hat_pad[..., :H0, :W0]     # crop to (B,1,H0,W0)
        alpha = alpha  [..., :H0, :W0]


        assert_finite(d0, "d0")
        assert_finite(c0, "c0")
        assert_finite(d_tex, "d_tex after UNet")
        assert_finite(d_sem, "d_sem after UNet")
        assert_finite(alpha, "alpha")
        assert_finite(d_hat_pad, "final output d_hat_pad")

        return {
            "depth"        : d_hat,
            "coarse_depth" : d0.unsqueeze(0),
            "confidence"   : c0.unsqueeze(0),
            "alpha"        : alpha,
        }


    def get_trainable_parameters(self):
        return itertools.chain(
            self.tex_unet.parameters(),          # ← full-res residual
            self.sem_unet.parameters(),          # ← semantic residual
            self.gate_conv.parameters(),         # ← confidence gate
            self.semantic_adapter.parameters(),  # ← tiny MLP for CLS token
        )


def si_log_loss(d_pred, d_gt, mask=None):
    """Scale-invariant log RMSE (Eigen et al.)."""
    eps   = 1e-6
    log_d = torch.log(d_pred.clamp(min=eps))
    log_g = torch.log(d_gt.clamp(min=eps))
    if mask is not None:
        log_d, log_g = log_d[mask], log_g[mask]
    diff  = log_d - log_g
    return torch.sqrt(torch.mean(diff**2) - torch.mean(diff)**2)

def si_log_loss_per_image(d_pred, d_gt, mask=None, reduction='mean'):
    """
    Compute scale-invariant log-RMSE per image, then reduce over batch.
    
    d_pred, d_gt:  tensors of shape [B, ...]
    mask:         optional bool tensor of same shape, applied per-sample
    reduction:    'mean' | 'sum' | 'none'
    
    Returns:
      If reduction='none', returns a [B] tensor of per-image losses;
      otherwise returns a scalar.
    """
    eps = 1e-6
    B = d_pred.shape[0]
    losses = []
    for i in range(B):
        dp = d_pred[i].clamp(min=eps).log()
        dg = d_gt[i].clamp(min=eps).log()
        if mask is not None:
            m = mask[i]
            dp, dg = dp[m], dg[m]
        diff = dp - dg
        # per-image scale-invariant RMSE
        losses.append(torch.sqrt(diff.pow(2).mean() - diff.mean().pow(2)))
    losses = torch.stack(losses, dim=0)  # [B]
    
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:  # 'none'
        return losses

# class RGBDepthPairs(Dataset):
#     """
#     Yields (rgb_tensor, gt_depth_tensor, path_str) for training / val.
#     rgb  :  float32,  [0,1],  shape (3,H,W)
#     depth:  float32,  metres, shape (1,H,W)
#     """
#     def __init__(self, root: Path, list_txt: Path):
#         self.root = Path(root)
#         self.pairs = [l.strip().split() for l in Path(list_txt).read_text().splitlines() if l.strip()]

#     def __len__(self):            return len(self.pairs)

#     def __getitem__(self, idx):
#         rgb_file, depth_file = self.pairs[idx]
#         # RGB
#         rgb = Image.open(self.root / rgb_file).convert("RGB")
#         rgb = TF.to_tensor(rgb)                         # (3,H,W) float32 [0,1]
#         # GT depth
#         # Load depth if it exists
#         depth_path = self.root / depth_file
#         if depth_path.exists():
#             depth = np.load(depth_path).astype(np.float32)
#             depth = torch.from_numpy(depth)[None]  # (1, H, W)
#             return rgb, depth, str(self.root / rgb_file)
#         else:
#             depth = None
#             return rgb, str(self.root / rgb_file)


class RGBDepthPairs(Dataset):
    def __init__(self, root: Path, list_txt: Path):
        list_txt = Path(list_txt)
        self.root      = Path(root)
        self.cache_dir = Path("/local/home/idemir/Desktop/depth-cil/monocular_depth_estimation/data/cached_vggt_raw_depths_and_conf/train/cache_coarse")
        self.pairs = [l.strip().split() for l in list_txt.read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_file, depth_file = self.pairs[idx]
        rgb = Image.open(self.root / rgb_file).convert("RGB")
        rgb = TF.to_tensor(rgb)

        stem = Path(rgb_file).stem
        d0 = torch.from_numpy(
            np.load(self.cache_dir / f"{stem}.depth.npy")
        ).unsqueeze(0)  # (1,H,W)
        c0 = torch.from_numpy(
            np.load(self.cache_dir / f"{stem}.conf.npy")
        ).unsqueeze(0)

        return rgb, d0, c0, str(self.root / rgb_file)
    
    def __getitem__(self, idx):
        rgb_file, depth_file = self.pairs[idx]
        # RGB
        rgb = Image.open(self.root / rgb_file).convert("RGB")
        rgb = TF.to_tensor(rgb)                         # (3,H,W) float32 [0,1]
        # GT depth
        # Load depth if it exists
        depth_path = self.root / depth_file
        if depth_path.exists():
            stem = Path(rgb_file).stem
            d0 = torch.from_numpy(
                np.load(self.cache_dir / f"{stem}.depth.npy")
            ).unsqueeze(0)  # (1,H,W)
            c0 = torch.from_numpy(
                np.load(self.cache_dir / f"{stem}.conf.npy")
            ).unsqueeze(0)
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)[None]  # (1, H, W)
            return rgb, depth, str(self.root / rgb_file), d0, c0
        else:
            depth = None
            return rgb, str(self.root / rgb_file)




def joint_bilateral(depth, img):
    img = img.squeeze(0).permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
    depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
    img_np = img.detach().cpu().numpy().astype(np.float32)

    return cv2.ximgproc.jointBilateralFilter(
        img_np,  # joint/guide image
        depth_np,  # src image to be filtered
        9,  # diameter of each pixel neighborhood
        75,  # sigmaColor
        75,  # sigmaSpace
    )


@torch.no_grad()
def evaluate_metrics(pipe, loader, device='cuda', mixed_precision=True):
    """
    Runs one pass over loader and returns a dict of metrics,
    all properly scale-aligned (except si-log) and averaged per sample.
    """
    total_samples = 0
    sum_si, sum_mse, sum_mae, sum_d1, sum_d2 = 0.0, 0.0, 0.0, 0.0, 0.0

    for rgb, depth_gt, _, d0, c0 in tqdm(loader):
        bs = rgb.size(0)
        total_samples += bs

        rgb, depth_gt = rgb.to(device), depth_gt.to(device)

        with autocast(enabled=mixed_precision):
            out    = pipe.forward(image=rgb, img_path=None, d0=d0, c0=c0)
            d_pred = out['depth']  # (B, H, W)

        # Ensure no zero or NaNs in medians
        eps = 1e-6
        med_pred = torch.median(d_pred.view(bs, -1), dim=1).values.clamp(min=eps)
        med_gt   = torch.median(depth_gt.view(bs, -1), dim=1).values.clamp(min=eps)
        scaling  = (med_gt / med_pred).view(-1, 1, 1)
        d_pred_scaled = d_pred * scaling  # (B, H, W)

        # --- compute per-batch metrics ---
        # scale-invariant log RMSE
        sil = si_log_loss(d_pred, depth_gt)

        # per-pixel error metrics
        mse = torch.mean((d_pred_scaled - depth_gt) ** 2)
        mae = torch.mean(torch.abs(d_pred_scaled - depth_gt))

        # threshold accuracy
        ratio = torch.max(d_pred_scaled / depth_gt, depth_gt / d_pred_scaled)
        δ1 = torch.mean((ratio < 1.25).float())
        δ2 = torch.mean((ratio < 1.25**2).float())

        # accumulate with batch-size weighting
        sum_si  += sil.item()  * bs
        sum_mse += mse.item()  * bs
        sum_mae += mae.item()  * bs
        sum_d1  += δ1.item()   * bs
        sum_d2  += δ2.item()   * bs

    # return per-sample averages
    return {
        'si_log': sum_si  / total_samples,
        'MSE'   : sum_mse / total_samples,
        'MAE'   : sum_mae / total_samples,
        'delta1': sum_d1  / total_samples,
        'delta2': sum_d2  / total_samples,
    }


def plot_loss_curves(train_hist, val_hist, out_path):
    """Simple train vs val si_log loss plot."""
    plt.figure()
    plt.plot(train_hist, label='Train si_log')
    plt.plot(val_hist,   label='Eval si_log')
    plt.xlabel('Epoch')
    plt.ylabel('si_log_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def train_refiner(
        pipe,
        train_loader,
        eval_loader,
        test_loader  = None,
        epochs       = 1000,
        lr           = 2e-4,
        weight_decay = 1e-2,
        mixed_precision = True,
        log_every    = 100,
        dump_every   = 5,
        # dump_root    = Path("unet_predictions"),
        # ckpt_dir     = Path("checkpoints"),
):

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_root = Path(f"dual_unet_exp_{timestamp}")
    dump_root = exp_root / "predictions"
    ckpt_dir  = exp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda'
    opt  = torch.optim.AdamW(pipe.get_trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=mixed_precision)

    running_refine, running_coarse = 0.0, 0.0 

    train_loss_history = []
    eval_loss_history = []

    for epoch in range(1, epochs + 1):
        total_loss_ref  = 0.0
        total_loss_coa  = 0.0
        total_samples   = 0
        global_step     = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch:02d}/{epochs}")

        for i, (rgb, depth_gt, path_str, d0, c0) in pbar:
            bs = rgb.size(0)
            rgb, depth_gt = rgb.to(device), depth_gt.to(device)

            with autocast(enabled=mixed_precision):
                out         = pipe.forward(image=rgb, img_path=path_str[0], d0=d0, c0=c0)
                d_refined   = out["depth"]
                d_coarse    = out["coarse_depth"].unsqueeze(1)  # ensure shape [B,1,H,W]

                loss_refine = si_log_loss_per_image(d_refined, depth_gt)
                loss_coarse = si_log_loss_per_image(d_coarse,  depth_gt)

            # backward & step
            scaler.scale(loss_refine).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            # accumulate *per-sample* loss
            total_loss_ref += loss_refine.item() * bs
            total_loss_coa += loss_coarse.item() * bs
            total_samples  += bs
            global_step   += 1

            # log averages every log_every steps
            if global_step % log_every == 0:
                avg_ref = total_loss_ref / total_samples
                avg_coa = total_loss_coa / total_samples
                pbar.set_postfix({
                  "μ_ref": f"{avg_ref:.4f}",
                  "μ_coa": f"{avg_coa:.4f}"
                })

            if total_samples >= 2000:
                break

        pbar.close()

        # compute final per-epoch average
        mean_ref  = total_loss_ref / total_samples
        mean_coa  = total_loss_coa / total_samples
        train_loss_history.append(mean_ref)
        mean_ref = 0.0
        metrics = evaluate_metrics(pipe, eval_loader,
                                   device=device,
                                   mixed_precision=mixed_precision)
        eval_loss_history.append(metrics['si_log'])

        print(f"\nEpoch {epoch:03d}/{epochs}  "
              f"Train SI-Log: {mean_ref:.4f}  "
              f"Eval  SI-Log: {metrics['si_log']:.4f}  "
              f"MSE: {metrics['MSE']:.4f}  "
              f"MAE: {metrics['MAE']:.4f}  "
              f"δ<1.25: {metrics['delta1']:.3f}  "
              f"δ<1.25²: {metrics['delta2']:.3f}\n")


        if (test_loader is not None) and (epoch % dump_every == 0):
            step_dir   = dump_root / f"step_{epoch:06d}"
            ref_dir    = step_dir / "refined"   
            coarse_dir = step_dir / "coarse"       

            ref_dir.mkdir(parents=True, exist_ok=True)
            coarse_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad(), autocast(enabled=mixed_precision):
                for rgb_t, path_t in tqdm(test_loader):
                    rgb_t = rgb_t.to(device)
                    fname = Path(path_t[0]).stem + ".npy"

                    out  = pipe.forward(image=rgb_t, img_path=path_t[0])

                    pred = out["depth"][0, 0]  # (H,W)
                    pred = joint_bilateral(pred, rgb_t)
                    np.save(ref_dir / fname, pred)

            torch.save({
                "epoch": epoch,
                "tex_unet": pipe.tex_unet.state_dict(),
                "sem_unet": pipe.sem_unet.state_dict(),
                "gate_conv": pipe.gate_conv.state_dict(),
                "semantic_adapter": pipe.semantic_adapter.state_dict(),
                "opt": opt.state_dict(),
            }, ckpt_dir / f"refiner_ep{epoch:06d}.pt")
            print("✓ saved checkpoint")

            print(f"Dumped test predictions to {step_dir}")

        plot_loss_curves(
            train_loss_history,
            eval_loss_history,
            out_path=exp_root / f"loss_curves_{epoch}.png"
        )


def load_refiner_for_eval(pipe: DepthRefinementPipeline, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    pipe.tex_unet.load_state_dict(ckpt["tex_unet"])
    pipe.sem_unet.load_state_dict(ckpt["sem_unet"])
    pipe.gate_conv.load_state_dict(ckpt["gate_conv"])
    pipe.semantic_adapter.load_state_dict(ckpt["semantic_adapter"])
    print("✓ Loaded model weights for evaluation")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"



    # Set a seed for reproducibility
    torch.manual_seed(1881)
    train_ds = RGBDepthPairs(root="data/train/train", list_txt="data/train_list.txt")
    test_ds   = RGBDepthPairs(root="data/test/test",  list_txt="data/test_list.txt")
    eval_size = int(len(train_ds) * 0.2)
    new_train_ds, eval_ds = random_split(train_ds, [len(train_ds) - eval_size, eval_size])
 
    new_train_loader = DataLoader(new_train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    pipe = DepthRefinementPipeline().to(device)
    pipe.clip_model = pipe.clip_model.to(device)
    pipe.semantic_adapter = pipe.semantic_adapter.to(device)
    pipe.tex_unet = pipe.tex_unet.to(device)
    pipe.sem_unet = pipe.sem_unet.to(device)
    pipe.gate_conv = pipe.gate_conv.to(device)

    # load_refiner_for_eval(pipe, Path("/local/home/idemir/Desktop/depth-cil/monocular_depth_estimation/dual_unet_exp_20250529_2023/checkpoints/refiner_ep000015.pt"))

    train_refiner(pipe,
                  train_loader=new_train_loader,
                  eval_loader=new_train_loader,
                  test_loader=test_loader,
                  epochs=1000,
                  lr=2e-4)