import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm

from model import BrainTumorSegmentationModel, SimpleBSCLoss
from utils import create_synthetic_brain_data, save_checkpoint, TrainingLogger, calculate_metrics, visualize_predictions
from synthetic_loader import SyntheticBrainDataset, RandomAugmentation
from brats_loader import BraTS2020Dataset  # Add this import

class SyntheticBrainDatasetOld(Dataset):
    """Synthetic brain tumor dataset for testing (old version - generates on the fly)"""
    def __init__(self, num_samples=100, depth=32, height=64, width=64):
        self.num_samples = num_samples
        self.depth = depth
        self.height = height
        self.width = width
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        data, target = create_synthetic_brain_data(1, self.depth, self.height, self.width)
        return data[0], target[0]  # Remove batch dimension

def create_data_loader(args, split='train'):
    """Create appropriate data loader based on dataset type"""
    if args.dataset == 'synthetic_saved':
        # Use pre-saved synthetic data
        dataset = SyntheticBrainDataset(split=split, transform=RandomAugmentation() if args.augment else None)
        print(f"Using pre-saved synthetic data: {len(dataset)} {split} samples")
    
    elif args.dataset == 'synthetic_old':
        # Use on-the-fly generated data (old method)
        dataset = SyntheticBrainDatasetOld(num_samples=args.num_samples)
        print(f"Using on-the-fly generated synthetic data: {args.num_samples} samples")
    
    elif args.dataset == 'brats':
        # Use BraTS 2020 data
        dataset = BraTS2020Dataset(args.brats_path, split=split)
        print(f"Using BraTS 2020 data: {len(dataset)} {split} samples")
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=(split=='train'))

def train_model(args):
    """Main training function"""
    print("🚀 Starting Training...")
    print(f"📊 Dataset: {args.dataset}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = BrainTumorSegmentationModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.base_channels
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Loss function and optimizer
    criterion = SimpleBSCLoss(alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create data loaders
    train_loader = create_data_loader(args, 'train')
    
    # For BraTS, create validation loader
    if args.dataset == 'brats':
        val_loader = create_data_loader(args, 'val')
    
    # Logger
    logger = TrainingLogger()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_dice_scores = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            dice_scores = calculate_metrics(output, target)
            
            epoch_loss += loss.item()
            epoch_dice_scores.append(dice_scores)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{np.mean(dice_scores):.4f}'
            })
        
        # Average metrics for epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = np.mean(epoch_dice_scores, axis=0)
        
        # Log epoch
        logger.log_epoch(epoch, avg_loss, avg_dice.tolist())
        
        print(f'Epoch {epoch+1}/{args.epochs}: Loss: {avg_loss:.4f}, '
              f'Dice: {avg_dice[0]:.4f}/{avg_dice[1]:.4f}/{avg_dice[2]:.4f}')
        
        # Validation for BraTS
        if args.dataset == 'brats' and (epoch + 1) % 5 == 0:
            val_loss, val_dice = validate_model(model, val_loader, criterion, device)
            print(f'Validation - Loss: {val_loss:.4f}, Dice: {val_dice[0]:.4f}/{val_dice[1]:.4f}/{val_dice[2]:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"checkpoints/{args.dataset}_model_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
    
    # Save final model
    torch.save(model.state_dict(), f"checkpoints/{args.dataset}_final_model.pth")
    print("✅ Training completed!")
    
    # Plot training progress
    logger.plot_training(f"outputs/{args.dataset}_training_progress.png")
    
    return model

def validate_model(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0.0
    val_dice_scores = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            val_loss += loss.item()
            
            dice_scores = calculate_metrics(output, target)
            val_dice_scores.append(dice_scores)
    
    model.train()
    return val_loss / len(val_loader), np.mean(val_dice_scores, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, default=4, help='Input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='Output channels')
    parser.add_argument('--base_channels', type=int, default=16, help='Base channels')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='synthetic_saved', 
                       choices=['synthetic_saved', 'synthetic_old', 'brats'],
                       help='Dataset to use for training')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples')
    parser.add_argument('--brats_path', type=str, default='data/BraTS2020', help='Path to BraTS 2020 data')
    
    # Other parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='BSC loss weight')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Train model
    model = train_model(args)

if __name__ == "__main__":
    main()