import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from data.loader import DepthDataset, get_dataloaders, get_transforms
from utils.helpers import ensure_dir, target_transform, load_config, aleatoric_loss, laplacian_aleatoric_loss, deep_supervision_loss, scale_invariant_laplacian_aleatoric_loss
from models.AleatoricUNet import SimpleUNet, AleatoricUNet
from models.DropoutUNet import DropoutUNet, DropoutUNet_ResNet50, DropoutUNet_ResNet101
from models.AttentionUNet import AttentionUNet, AttentionUNet_ResNet101
from train.trainer import train_model
from eval.eval import evaluate_model
from submission.submit import generate_test_predictions
from torch.optim.lr_scheduler import ReduceLROnPlateau

# configs
config = load_config()
data_dir = config.data.data_dir
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = config.train.learning_rate
NUM_EPOCHS = config.train.num_epochs
WEIGHT_DECAY = config.train.weight_decay
MC_TRAINING = config.train.mc_training
MC_DROPOUT = config.eval.mc_dropout
DROPOUT_P = config.model.dropout_p

# Main
#####

def main():
    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Clear CUDA cache before model initialization
    torch.cuda.empty_cache()
    
    # Fix random seeds for reproducibility
    torch.manual_seed(config.seed)

    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Initialize model
    # model = AttentionUNet_ResNet101(dropout_rate=DROPOUT_P)
    model = DropoutUNet_ResNet101(dropout_rate=DROPOUT_P)

    # Load weights for fine-tuning if specified
    if config.model.pretrained:
        pretrained_dir = os.path.join(data_dir, config.model.model_name)
        weights_dir = os.path.join(pretrained_dir, 'results/best_model_sirmse.pth')

        dropout_weights = torch.load(weights_dir)
        model_dict = model.state_dict()

        # Filter out keys that don’t match
        pretrained_dict = {k: v for k, v in dropout_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model_name = config.model.model_name
    # model_name = 'AttentionUNet-ResNet101-p02-finetuned'
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")


    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # define output directory
    output_dir = os.path.join(data_dir, model_name)
    results_dir = os.path.join(output_dir, 'results')
    predictions_dir = os.path.join(output_dir, 'predictions')

    # Create output directories
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)

    # Save the configuration used for training
    config_path = "configs/config.yaml" 
    destination_path = os.path.join(results_dir, "config.yaml")

    shutil.copyfile(config_path, destination_path)

    # Training parameters
    criterion = scale_invariant_laplacian_aleatoric_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if MC_TRAINING:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    # continue training from a checkpoint
    checkpoint_path = os.path.join(results_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    
    # use best sirmse model if available
    checkpoint_path = os.path.join(results_dir, 'checkpoint_epoch_5 (0.0741).pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    
    """
    # Train the model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, DEVICE, results_dir, mc_training=MC_TRAINING)
    
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    print("Loading the best model for evaluation...")
    
    """
    # Evaluate the model on validation set
    print("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE, results_dir, mc_predictions=MC_DROPOUT)

    # Print metrics
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save metrics to file
    with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    """
    sample_dir = os.path.join(results_dir, config.eval.sample_output_dir)
    ensure_dir(sample_dir)

    # Generate predictions for the test set
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE, predictions_dir, sample_output_dir=sample_dir, postprocess=config.eval.postprocess)
    
    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")
    """
if __name__ == '__main__':
    main()