# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import torch.nn.functional as F
from PIL import Image
import pickle

def create_synthetic_brain_data(batch_size=2, depth=32, height=64, width=64, num_modalities=4):
    """Create synthetic MRI-like data for testing"""
    # Create random 3D data resembling MRI scans
    data = np.random.randn(batch_size, num_modalities, depth, height, width).astype(np.float32)
    
    # Create simple synthetic tumor masks (3 classes)
    masks = np.zeros((batch_size, 3, depth, height, width), dtype=np.float32)
    
    # Add some spherical "tumors"
    for b in range(batch_size):
        # Create multiple tumors
        for tumor_idx in range(3):
            center_d = np.random.randint(8, depth-8)
            center_h = np.random.randint(16, height-16)
            center_w = np.random.randint(16, width-16)
            radius = np.random.randint(5, 12)
            
            for d in range(max(0, center_d-radius), min(depth, center_d+radius)):
                for h in range(max(0, center_h-radius), min(height, center_h+radius)):
                    for w in range(max(0, center_w-radius), min(width, center_w+radius)):
                        dist = np.sqrt((d-center_d)**2 + (h-center_h)**2 + (w-center_w)**2)
                        if dist < radius:
                            masks[b, 0, d, h, w] = 1.0  # Whole tumor
                        if dist < radius * 0.7:
                            masks[b, 1, d, h, w] = 1.0  # Tumor core
                        if dist < radius * 0.3:
                            masks[b, 2, d, h, w] = 1.0  # Enhancing tumor
    
    return torch.from_numpy(data), torch.from_numpy(masks)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {path}")
    return model, optimizer, epoch, loss

def visualize_predictions(data, target, prediction, save_path=None):
    """Visualize input, ground truth, and prediction with dimension handling"""
    # Resize prediction to match target dimensions if needed
    if prediction.shape[-3:] != target.shape[-3:]:
        prediction_resized = F.interpolate(prediction, size=target.shape[-3:], mode='trilinear', align_corners=True)
    else:
        prediction_resized = prediction
    
    # Take middle slice
    slice_idx = data.shape[2] // 2
    
    # Determine number of input modalities to display
    num_modalities = min(4, data.shape[1])  # Show up to 4 modalities
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Input modalities
    modalities = ['T1', 'T2', 'FLAIR', 'T1ce']
    for i in range(num_modalities):
        if i < data.shape[1]:  # Check if modality exists
            axes[0, i].imshow(data[0, i, slice_idx].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'Input: {modalities[i]}')
        else:
            axes[0, i].axis('off')  # Hide if modality doesn't exist
        axes[0, i].axis('off')
    
    # Hide any unused modality subplots
    for i in range(num_modalities, 4):
        axes[0, i].axis('off')
    
    # Ground Truth
    gt_combined = torch.argmax(target[0, :, slice_idx], dim=0).cpu().numpy()
    im1 = axes[1, 0].imshow(gt_combined, cmap='jet')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Prediction
    pred_combined = torch.argmax(prediction_resized[0, :, slice_idx], dim=0).cpu().numpy()
    im2 = axes[1, 1].imshow(pred_combined, cmap='jet')
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Difference
    diff = np.abs(gt_combined - pred_combined)
    im3 = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2])
    
    # Dice scores text
    dice_scores = calculate_metrics(prediction, target)
    axes[1, 3].text(0.1, 0.9, f'Dice Scores:', fontsize=12, fontweight='bold')
    axes[1, 3].text(0.1, 0.7, f'WT: {dice_scores[0]:.4f}', fontsize=10)
    axes[1, 3].text(0.1, 0.5, f'TC: {dice_scores[1]:.4f}', fontsize=10)
    axes[1, 3].text(0.1, 0.3, f'ET: {dice_scores[2]:.4f}', fontsize=10)
    axes[1, 3].text(0.1, 0.1, f'Avg: {np.mean(dice_scores):.4f}', fontsize=10, fontweight='bold')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    else:
        plt.show()

def calculate_metrics(prediction, target):
    """Calculate segmentation metrics with dimension handling"""
    # Resize prediction to match target dimensions if needed
    if prediction.shape[-3:] != target.shape[-3:]:
        prediction_resized = F.interpolate(prediction, size=target.shape[-3:], mode='trilinear', align_corners=True)
    else:
        prediction_resized = prediction
    
    prediction_bin = (prediction_resized > 0.5).float()
    target_bin = (target > 0.5).float()
    
    # Dice coefficient for each class
    dice_scores = []
    for class_idx in range(prediction.shape[1]):
        # Flatten spatial dimensions for proper calculation
        pred_flat = prediction_bin[:, class_idx].contiguous().view(prediction_bin.shape[0], -1)
        target_flat = target_bin[:, class_idx].contiguous().view(target_bin.shape[0], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection) / (union + 1e-8)
        dice_scores.append(dice.mean().item())
    
    return dice_scores

class TrainingLogger:
    """Simple training logger"""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.log_data = {
            'loss': [],
            'dice_scores': [],
            'epochs': []
        }
    
    def log_epoch(self, epoch, loss, dice_scores):
        """Log epoch results"""
        self.log_data['epochs'].append(epoch)
        self.log_data['loss'].append(loss)
        self.log_data['dice_scores'].append(dice_scores)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def plot_training(self, save_path=None):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.log_data['epochs'], self.log_data['loss'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # Plot dice scores
        dice_scores = np.array(self.log_data['dice_scores'])
        for i in range(dice_scores.shape[1]):
            ax2.plot(self.log_data['epochs'], dice_scores[:, i], label=f'Class {i}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Dice Scores')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

def save_synthetic_dataset(num_samples=1000, split_ratio=(0.7, 0.2, 0.1)):
    """
    Generate and save a large synthetic dataset to disk
    """
    import os
    import pickle
    
    # Create directories
    base_dir = "data/synthetic"
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
    
    # Calculate split sizes
    train_size = int(num_samples * split_ratio[0])
    val_size = int(num_samples * split_ratio[1])
    test_size = num_samples - train_size - val_size
    
    print(f"🎲 Generating synthetic dataset: {train_size} train, {val_size} val, {test_size} test")
    
    # Generate and save training data
    for i in range(train_size):
        data, target = create_synthetic_brain_data(batch_size=1, depth=32, height=128, width=128)
        
        # Save as pickle files
        sample = {
            'data': data,
            'target': target,
            'patient_id': f"synth_train_{i:04d}",
            'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
            'tumor_classes': ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
        }
        
        with open(os.path.join(base_dir, 'train', f'sample_{i:04d}.pkl'), 'wb') as f:
            pickle.dump(sample, f)
    
    # Generate and save validation data
    for i in range(val_size):
        data, target = create_synthetic_brain_data(batch_size=1, depth=32, height=128, width=128)
        
        sample = {
            'data': data,
            'target': target,
            'patient_id': f"synth_val_{i:04d}",
            'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
            'tumor_classes': ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
        }
        
        with open(os.path.join(base_dir, 'val', f'sample_{i:04d}.pkl'), 'wb') as f:
            pickle.dump(sample, f)
    
    # Generate and save test data
    for i in range(test_size):
        data, target = create_synthetic_brain_data(batch_size=1, depth=32, height=128, width=128)
        
        sample = {
            'data': data,
            'target': target,
            'patient_id': f"synth_test_{i:04d}",
            'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
            'tumor_classes': ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
        }
        
        with open(os.path.join(base_dir, 'test', f'sample_{i:04d}.pkl'), 'wb') as f:
            pickle.dump(sample, f)
    
    # Save dataset info
    dataset_info = {
        'total_samples': num_samples,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'image_shape': (32, 128, 128),
        'modalities': 4,
        'classes': 3,
        'generated_date': str(np.datetime64('now'))
    }
    
    with open(os.path.join(base_dir, 'dataset_info.pkl'), 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"✅ Synthetic dataset saved to {base_dir}")
    print(f"📊 Dataset info: {dataset_info}")

def create_sample_images(num_images=10):
    """
    Create sample synthetic images for demo purposes
    """
    import os
    from PIL import Image
    
    samples_dir = "data/synthetic/samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(num_images):
        data, target = create_synthetic_brain_data(batch_size=1, depth=32, height=256, width=256)
        
        # Save input (FLAIR modality) as PNG
        input_slice = data[0, 2, 16].numpy()  # Middle slice, FLAIR channel
        input_img = Image.fromarray((input_slice * 255).astype(np.uint8))
        input_img.save(os.path.join(samples_dir, f'input_{i:02d}.png'))
        
        # Save ground truth as PNG
        gt_slice = torch.argmax(target[0, :, 16], dim=0).numpy()
        gt_img = Image.fromarray((gt_slice * 85).astype(np.uint8))  # Scale for visibility
        gt_img.save(os.path.join(samples_dir, f'ground_truth_{i:02d}.png'))
    
    print(f"✅ Sample images saved to {samples_dir}")

if __name__ == "__main__":
    # Test the utils
    data, target = create_synthetic_brain_data()
    print(f"Synthetic data shape: {data.shape}")
    print(f"Synthetic target shape: {target.shape}")
    print("Utils module working correctly!")