from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict
import math
import cv2
from PIL import Image

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
            out_channels=2,
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
            out_channels=2,
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
        self.gate_conv = gate_conv or nn.Conv2d(4, 1, 3, padding=1)

        # nn.init.constant_(self.gate_conv.bias, 0)
        # nn.init.constant_(self.gate_conv.weight, 0)

        self.register_modules(
            clip_model=self.clip_model,
            semantic_adapter=self.semantic_adapter,
            tex_unet=self.tex_unet,
            sem_unet=self.sem_unet,
            gate_conv=self.gate_conv,
        )


    def forward(self, *, image: Tensor, img_path: str) -> Dict[str, Tensor]:
        device, dtype = image.device, image.dtype
        B, _, H0, W0 = image.shape 

        # ------------------------------------------------------------
        # 1. Coarse VGGT prediction
        # ------------------------------------------------------------
        d0, c0 = vggt_get_depth(img_path, H0, W0)      #  (H0, W0)
        d0, c0 = d0.to(device=device, dtype=dtype), c0.to(device=device, dtype=dtype)
        sigma_c0 = torch.sigmoid(c0)

        self.clip_model = self.clip_model.to(device)
        self.semantic_adapter = self.semantic_adapter.to(device)
        self.tex_unet = self.tex_unet.to(device)
        self.sem_unet = self.sem_unet.to(device)
        self.gate_conv = self.gate_conv.to(device)
        # ------------------------------------------------------------
        # 2. Global semantics (CLS token)
        # ------------------------------------------------------------
        clip_in = self.clip_processor(
            images=(image * 255).clamp(0, 255).to(torch.uint8),
            return_tensors="pt"
        ).pixel_values.to(device)
        cls      = self.clip_model(pixel_values=clip_in).last_hidden_state[:, 0]  # (B,768)
        sem_vec  = self.semantic_adapter(cls)                                     # (B,64)

        # ------------------------------------------------------------
        # 3. Build 5-channel core tensor and pad
        # ------------------------------------------------------------
        x_core = torch.cat((image, d0.unsqueeze(0).unsqueeze(0), sigma_c0.unsqueeze(0).unsqueeze(0)), dim=1)  # (B,5,H0,W0)

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
        d_tex_raw = self.tex_unet(sample=x_core, timestep=t0).sample
        d_tex, c_tex = d_tex_raw[:, :1], torch.sigmoid(d_tex_raw[:, 1:2])                # (B,1,H_pad,W_pad)

        # ------------------------------------------------------------
        # 5. Semantic branch (¼ res, cross-attn)
        # ------------------------------------------------------------
        x_sem_low = F.interpolate(x_core, scale_factor=0.25, mode="bilinear", align_corners=False)
        d_sem_raw = self.sem_unet(
            sample=x_sem_low,
            timestep=t0,
            encoder_hidden_states=sem_vec.unsqueeze(1)   # (B,1,64)  seq_len=1
        ).sample                                         # (B,1,H_pad/4,W_pad/4)

        d_sem, c_sem = d_sem_raw[:, :1], torch.sigmoid(d_sem_raw[:, 1:2])
        conf_reg = (c_tex.mean() - 0.5).abs() + (c_sem.mean() - 0.5).abs()
        d_sem = F.interpolate(d_sem, size=(H_pad, W_pad), mode="bilinear", align_corners=False)
        c_sem = F.interpolate(c_sem, size=(H_pad, W_pad), mode="bilinear", align_corners=False)

        if not torch.isfinite(d_sem).all():
            print("d_sem has inf or NaN")
        if not torch.isfinite(d_tex).all():
            print("d_tex has inf or NaN")

        # def normalize_residual(res: torch.Tensor, eps=1e-6) -> torch.Tensor:
        #     """Normalize to zero mean, unit std per batch."""
        #     mean = res.mean(dim=[2, 3], keepdim=True)
        #     std  = res.std(dim=[2, 3], keepdim=True).clamp(min=eps)
        #     return (res - mean) / std

        # # Normalize before scaling
        # d_sem_norm = normalize_residual(d_sem)
        # d_tex_norm = normalize_residual(d_tex)

        max_change = d0.abs().max() * 0.1  # 10% of coarse depth range
        d_sem = max_change * torch.tanh(d_sem)
        d_tex = max_change * torch.tanh(d_tex)

        # ------------------------------------------------------------
        # 6. Confidence-aware fusion
        # ------------------------------------------------------------
        fusion_input = torch.cat((d_tex, c_tex, d_sem, c_sem), dim=1)
        alpha = torch.sigmoid(self.gate_conv(fusion_input))   # (B,1,H_pad,W_pad)
        
        d_tex_crop  = d_tex [..., :H0, :W0]
        d_sem_crop  = d_sem [..., :H0, :W0]
        alpha_crop  = alpha [..., :H0, :W0]
        d_hat_pad = d0.unsqueeze(0).unsqueeze(0) + alpha_crop * d_tex_crop + (1 - alpha_crop) * d_sem_crop         # (B,1,H_pad,W_pad)

        # ------------------------------------------------------------
        # 7. Remove padding, return native resolution
        # ------------------------------------------------------------
        d_hat = d_hat_pad[..., :H0, :W0]     # crop to (B,1,H0,W0)
        alpha = alpha  [..., :H0, :W0]

        return {
            "depth"        : d_hat,
            "coarse_depth" : d0.unsqueeze(0),
            "confidence"   : c0.unsqueeze(0),
            "alpha"        : alpha,
            "conf_reg"     : conf_reg,
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



class RGBDepthPairs(Dataset):
    """
    Yields (rgb_tensor, gt_depth_tensor, path_str) for training / val.
    rgb  :  float32,  [0,1],  shape (3,H,W)
    depth:  float32,  metres, shape (1,H,W)
    """
    def __init__(self, root: Path, list_txt: Path):
        self.root = Path(root)
        self.pairs = [l.strip().split() for l in Path(list_txt).read_text().splitlines() if l.strip()]

    def __len__(self):            return len(self.pairs)

    def __getitem__(self, idx):
        rgb_file, depth_file = self.pairs[idx]
        # RGB
        rgb = Image.open(self.root / rgb_file).convert("RGB")
        rgb = TF.to_tensor(rgb)                         # (3,H,W) float32 [0,1]
        # GT depth
        # Load depth if it exists
        depth_path = self.root / depth_file
        if depth_path.exists():
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)[None]  # (1, H, W)
            return rgb, depth, str(self.root / rgb_file)
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


def train_refiner(
        pipe,
        train_loader,
        test_loader  = None,
        epochs       = 1000,
        lr           = 2e-4,
        weight_decay = 1e-2,
        mixed_precision = True,
        log_every    = 20,
        dump_every   = 2500,
        dump_root    = Path("unet_predictions"),
        ckpt_dir     = Path("checkpoints"),
):
    device = 'cuda'
    opt  = torch.optim.AdamW(pipe.get_trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=mixed_precision)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    running_refine, running_coarse = 0.0, 0.0 

    for epoch in range(1, epochs + 1):
        for i, (rgb, depth_gt, path_str) in enumerate(train_loader):
            rgb, depth_gt = rgb.to(device), depth_gt.to(device)

            with autocast(enabled=mixed_precision):
                out          = pipe.forward(image=rgb, img_path=path_str[0])
                d_refined    = out["depth"]
                d_coarse     = out['coarse_depth'].unsqueeze(0)

                loss_refine  = si_log_loss(d_refined, depth_gt)
                loss_refine_reg = loss_refine + 0.5 * out['conf_reg']  # regularization term
                loss_coarse  = si_log_loss(d_coarse,  depth_gt)

            scaler.scale(loss_refine_reg).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        
            running_refine += loss_refine.item()
            running_coarse += loss_coarse.item()

            if (global_step) % log_every == 0:
                mean_refine  = running_refine / log_every
                mean_coarse  = running_coarse / log_every
                print(f"[{epoch:02d}/{epochs}] "
                    f"step {global_step:05d} "
                    f"loss_refine {loss_refine.item():.4f} "
                    f"loss_coarse {loss_coarse.item():.4f} "
                    f"Δ {loss_coarse.item() - loss_refine.item():.4f} "
                    f"| mean_refine {mean_refine:.4f}  "
                    f"mean_coarse {mean_coarse:.4f}  "
                    f"Δ̄ {mean_coarse - mean_refine:.4f}")

                running_refine = running_coarse = 0.0 

            global_step += 1


            if (test_loader is not None) and (global_step % dump_every == 0):
                step_dir   = dump_root / f"step_{global_step:06d}"
                ref_dir    = step_dir / "refined"   
                coarse_dir = step_dir / "coarse"       

                ref_dir.mkdir(parents=True, exist_ok=True)
                coarse_dir.mkdir(parents=True, exist_ok=True)
                with torch.no_grad(), autocast(enabled=mixed_precision):
                    for rgb_t, path_t in test_loader:
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
                }, ckpt_dir / f"refiner_ep{i:06d}.pt")
                print("✓ saved checkpoint")

                print(f"Dumped test predictions to {step_dir}")


        # torch.save({"epoch": epoch,
        #             "model": pipe.state_dict(),
        #             "opt"  : opt.state_dict()},
        #            ckpt_dir / f"refiner_ep{epoch:02d}.pt")
        # print("✓ saved checkpoint")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    root = Path("~/Desktop/MonocularDepth/ethz-cil-monocular-depth-estimation-2025").expanduser()
    train_path = os.path.join(root, "train/train")
    test_path  = os.path.join(root, "test/test")
    train_list = os.path.join(root, "train_list.txt")
    test_list  = os.path.join(root, "test_list.txt")

    train_ds = RGBDepthPairs(root=train_path, list_txt=train_list)
    test_ds   = RGBDepthPairs(root=test_path,     list_txt=test_list)
    """
    
    train_ds = RGBDepthPairs(root="data/train/train", list_txt="data/train_list.txt")
    test_ds   = RGBDepthPairs(root="data/test/test",     list_txt="data/test_list.txt")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    pipe = DepthRefinementPipeline().to(device)
    train_refiner(pipe,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  epochs=1,
                  lr=2e-5)