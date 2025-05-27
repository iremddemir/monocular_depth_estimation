import torch
import torch.nn as nn
from tqdm import tqdm
from utils.helpers import ensure_dir
import os
import numpy as np
import random
from utils.helpers import load_config, predict_mc_dropout, visualize_prediction, postprocess_depth

config = load_config()

def generate_test_predictions(model, test_loader, device, predictions_dir, sample_output_dir=None, postprocess=True):
    """Generate predictions for the test set without ground truth and optionally save 50 samples with uncertainty maps."""
    model.eval()
    ensure_dir(predictions_dir)

    if sample_output_dir:
        ensure_dir(sample_output_dir)
        saved_samples = set()
        max_samples = 50

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            if config.eval.mc_dropout:
                outputs, epistemic_uncertainty = predict_mc_dropout(model, inputs, num_samples=config.eval.mc_samples)
                mean_depth = outputs[:, 0:1, :, :]
                aleatoric_uncertainty = total_uncertainty = None

            elif config.model.deep_supervision:
                outputs, _ = model(inputs)
                mean_depth = outputs[:, 0:1, :, :]
                aleatoric_uncertainty = outputs[:, 1:2, :, :]
                epistemic_uncertainty = total_uncertainty = None

            else:
                outputs = model(inputs)
                mean_depth = outputs[:, 0:1, :, :]
                aleatoric_uncertainty = epistemic_uncertainty = total_uncertainty = None

            # Resize to original input dimensions (426x560)
            mean_depth = nn.functional.interpolate(mean_depth, size=(426, 560), mode='bilinear', align_corners=True)
            if aleatoric_uncertainty is not None:
                aleatoric_uncertainty = nn.functional.interpolate(aleatoric_uncertainty, size=(426, 560), mode='bilinear', align_corners=True)
            if epistemic_uncertainty is not None:
                epistemic_uncertainty = nn.functional.interpolate(epistemic_uncertainty, size=(426, 560), mode='bilinear', align_corners=True)
            if total_uncertainty is not None:
                total_uncertainty = nn.functional.interpolate(total_uncertainty, size=(426, 560), mode='bilinear', align_corners=True)

            for i in range(batch_size):
                filename = filenames[i].split(' ')[1]  # e.g., "123.npy"
                save_path = os.path.join(predictions_dir, f"{filename}")

                # Convert and postprocess depth
                depth_pred = mean_depth[i].cpu().squeeze().numpy()
                unc_map = None
                if postprocess:
                    if aleatoric_uncertainty is not None:
                        unc_map = aleatoric_uncertainty[i].cpu().squeeze().numpy()
                    elif epistemic_uncertainty is not None:
                        unc_map = epistemic_uncertainty[i].cpu().squeeze().numpy()

                    # Provide RGB as np.ndarray if available
                    rgb_np = inputs[i].cpu().numpy() if inputs.shape[1] == 3 else None
                    depth_pred = postprocess_depth(depth_pred, rgb=rgb_np, uncertainty=unc_map)

                np.save(save_path, depth_pred)

                # Save a sample with uncertainty maps if required
                if sample_output_dir and len(saved_samples) < max_samples:
                    if random.random() < 0.05:
                        base_name = os.path.splitext(filename)[0]

                        # Prepare data for visualization
                        depth_for_vis = depth_pred
                        uncertainties = {}
                        if epistemic_uncertainty is not None:
                            uncertainties['epistemic'] = epistemic_uncertainty[i].cpu().squeeze().numpy()
                        if aleatoric_uncertainty is not None:
                            uncertainties['aleatoric'] = aleatoric_uncertainty[i].cpu().squeeze().numpy()
                        if total_uncertainty is not None:
                            uncertainties['total'] = total_uncertainty[i].cpu().squeeze().numpy()

                        rgb = inputs[i] if inputs.shape[1] == 3 else None
                        visualize_prediction(
                            rgb=rgb,
                            depth=depth_for_vis,
                            uncertainties=uncertainties,
                            save_path=os.path.join(sample_output_dir, f"{base_name}.png")
                        )

                        saved_samples.add(base_name)


            del inputs, outputs
        torch.cuda.empty_cache()
