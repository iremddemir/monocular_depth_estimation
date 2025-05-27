import torch
from tqdm import tqdm
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.helpers import load_config, predict_mc_dropout, ensure_dir

# Load configuration
config = load_config()
mc_samples = config.eval.mc_samples
deep_supervision = config.model.deep_supervision

def evaluate_model(model, val_loader, device, results_dir, mc_predictions=False):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()
    
    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0
    
    total_samples = 0
    target_shape = None
    save_uncertainty_maps = getattr(config.eval, "save_uncertainty_maps", False)

    with torch.no_grad():
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating", dynamic_ncols=True, disable=not sys.stdout.isatty()):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape

            # Forward pass
            if mc_predictions:
                outputs, epistemic_uncertainty = predict_mc_dropout(model, inputs, num_samples=mc_samples)
                aleatoric_uncertainty = None
            elif deep_supervision:
                outputs, _ = model(inputs)
                epistemic_uncertainty = None
                aleatoric_uncertainty = outputs[:, 1:2, :, :]
            else:
                outputs = model(inputs)
                epistemic_uncertainty = None
                aleatoric_uncertainty = outputs[:, 1:2, :, :]

            # Extract mean prediction and interpolate to match target size
            mean_pred = outputs[:, 0:1, :, :]
            mean_pred = nn.functional.interpolate(mean_pred, size=targets.shape[-2:], mode='bilinear', align_corners=True)
            
            # Resize uncertainties if present
            if aleatoric_uncertainty is not None:
                aleatoric_uncertainty = nn.functional.interpolate(
                    aleatoric_uncertainty, size=targets.shape[-2:], mode='bilinear', align_corners=True
                )
            if epistemic_uncertainty is not None:
                epistemic_uncertainty = nn.functional.interpolate(
                    epistemic_uncertainty, size=targets.shape[-2:], mode='bilinear', align_corners=True
                )

            # Calculate metrics
            abs_diff = torch.abs(mean_pred - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()
            
            # Scale-invariant RMSE
            for i in range(batch_size):
                pred_np = mean_pred[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()
                EPSILON = 1e-6

                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue

                target_valid = target_np[valid_target]
                pred_valid = np.where(pred_np[valid_target] > EPSILON, pred_np[valid_target], EPSILON)

                diff = np.log(pred_valid) - np.log(target_valid)
                diff_mean = np.mean(diff)
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))

            max_ratio = torch.max(mean_pred / (targets + 1e-6), targets / (mean_pred + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            
            # Save predictions and uncertainty maps
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = mean_pred[i].cpu().squeeze().numpy()

                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)

                    plt.figure(figsize=(20, 5))
                    plt.subplot(1, 4, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis('off')

                    plt.subplot(1, 4, 2)
                    plt.imshow(target_np, cmap='plasma')
                    plt.title("Ground Truth")
                    plt.axis('off')

                    plt.subplot(1, 4, 3)
                    plt.imshow(output_np, cmap='plasma')
                    plt.title("Predicted")
                    plt.axis('off')

                    if save_uncertainty_maps:
                        total_uncertainty = None
                        plt.subplot(1, 4, 4)
                        if epistemic_uncertainty is not None and aleatoric_uncertainty is not None:
                            total_uncertainty = epistemic_uncertainty[i] + torch.exp(aleatoric_uncertainty[i])
                            plt.imshow(total_uncertainty.cpu().squeeze().numpy(), cmap='viridis')
                            plt.title("Total Uncertainty")
                        elif epistemic_uncertainty is not None:
                            plt.imshow(epistemic_uncertainty[i].cpu().squeeze().numpy(), cmap='viridis')
                            plt.title("Epistemic Uncertainty")
                        elif aleatoric_uncertainty is not None:
                            plt.imshow(torch.exp(aleatoric_uncertainty[i]).cpu().squeeze().numpy(), cmap='viridis')
                            plt.title("Aleatoric Uncertainty")
                        plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    plt.close()

        torch.cuda.empty_cache()

    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]
    metrics = {
        'MAE': mae / (total_samples * total_pixels),
        'RMSE': np.sqrt(rmse / (total_samples * total_pixels)),
        'siRMSE': sirmse / total_samples,
        'REL': rel / (total_samples * total_pixels),
        'Delta1': delta1 / (total_samples * total_pixels),
        'Delta2': delta2 / (total_samples * total_pixels),
        'Delta3': delta3 / (total_samples * total_pixels)
    }

    return metrics
