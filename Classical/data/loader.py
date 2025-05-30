import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.helpers import load_config, ensure_dir, target_transform, RandomFlipAndRotation, AddGaussianNoise, PairedTransform

# Load configuration
config = load_config()
INPUT_SIZE = config.data.input_size

BATCH_SIZE = config.data.batch_size
NUM_WORKERS = config.data.num_workers
PIN_MEMORY = config.data.pin_memory

# define paths
data_dir = config.data.data_dir
train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')

class DepthDataset(Dataset):
    def __init__(self, data_dir, list_file, transform=None, target_transform=None, paired_transform=None, has_gt=True):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.paired_transform = paired_transform
        self.has_gt = has_gt
        
        # Read file list
        with open(list_file, 'r') as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                # For test set without ground truth
                self.file_list = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)
    
    def __getitem__(self, idx):
        if self.has_gt:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])
            
            # Load RGB image and depth map
            rgb = Image.open(rgb_path).convert('RGB')
            depth = np.load(depth_path).astype(np.float32)

            # Apply paired transform if provided
            if self.paired_transform:
                rgb, depth = self.paired_transform(rgb, depth)

            # Apply individual transforms
            if self.transform:
                rgb = self.transform(rgb)
            if self.target_transform:
                depth = self.target_transform(depth)

            return rgb, depth, self.file_pairs[idx][0]  # Return filename for saving
        else:
            # Test set without ground truth
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(' ')[0])
            rgb = Image.open(rgb_path).convert('RGB')

            if self.transform:
                rgb = self.transform(rgb)
            
            return rgb, self.file_list[idx]

# Define transformations
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
            AddGaussianNoise(0, 0.01),  # Add Gaussian noise
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else: 
        return transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders():
    # Define transformations
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    paired_transform = PairedTransform()

    # Create training dataset with ground truth
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file, 
        transform=train_transform,
        target_transform=target_transform,
        paired_transform=paired_transform,
        has_gt=True
    )
    
    # Create test dataset without ground truth
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        transform=test_transform,
        has_gt=False  # Test set has no ground truth
    )

    # Split training dataset into train and validation
    val_split = config.data.validation_split
    total_size = len(train_full_dataset)
    train_size = int((1-val_split) * total_size)  
    val_size = total_size - train_size  
    
    # Set a fixed random seed for reproducibility
    torch.manual_seed(config.seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

   # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    return train_loader, val_loader, test_loader