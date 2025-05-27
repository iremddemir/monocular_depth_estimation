import torch
import torch.nn as nn
from tqdm import tqdm
from utils.helpers import ensure_dir
import os
import numpy as np
from utils.helpers import load_config, predict_mc_dropout

config = load_config()

def generate_test_predictions(model, test_loader, device, predictions_dir):
    """Generate predictions for the test set without ground truth"""
    model.eval()
    
    # Ensure predictions directory exists
    ensure_dir(predictions_dir)
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            if config.eval.mc_dropout:
                # Use MC Dropout for predictions
                outputs, epistemic_uncertainty = predict_mc_dropout(model, inputs, num_samples=config.eval.mc_samples)

            elif config.model.deep_supervision:
                # Forward pass with deep supervision
                outputs, _ = model(inputs)

            else:
                # Standard forward pass
                outputs = model(inputs)
            
            # Resize outputs to match original input dimensions (426x560)
            outputs = nn.functional.interpolate(
                outputs[:, 0:1, :, :],  # Only take the mean depth
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )
            
            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(' ')[1]
                
                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)
            
            # Clean up memory
            del inputs, outputs
        
        # Clear cache after test predictions
        torch.cuda.empty_cache()