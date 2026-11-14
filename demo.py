# demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import BrainTumorSegmentationModel
from utils import create_synthetic_brain_data, visualize_predictions

def run_demo():
    """Run a demo of the brain tumor segmentation model"""
    print("🎮 Starting Demo...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = BrainTumorSegmentationModel().to(device)
    
    # Load pretrained weights if available, otherwise use random initialization
    try:
        model.load_state_dict(torch.load('checkpoints/final_model.pth', map_location=device))
        print("✅ Loaded pretrained model")
    except:
        print("⚠️  No pretrained model found, using random initialization")
    
    # Create synthetic test data
    print("🧠 Generating synthetic brain MRI data...")
    test_data, test_target = create_synthetic_brain_data(batch_size=1, depth=32, height=64, width=64)
    test_data, test_target = test_data.to(device), test_target.to(device)
    
    # Run inference
    print("🔮 Running inference...")
    model.eval()
    with torch.no_grad():
        prediction = model(test_data)
    
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {prediction.shape}")
    print(f"Target shape: {test_target.shape}")
    
    # Calculate metrics
    from utils import calculate_metrics
    dice_scores = calculate_metrics(prediction, test_target)
    
    print(f"📊 Segmentation Results:")
    print(f"   Whole Tumor (WT) Dice: {dice_scores[0]:.4f}")
    print(f"   Tumor Core (TC) Dice: {dice_scores[1]:.4f}")
    print(f"   Enhancing Tumor (ET) Dice: {dice_scores[2]:.4f}")
    print(f"   Average Dice: {np.mean(dice_scores):.4f}")
    
    # Visualize results
    print("🎨 Generating visualization...")
    visualize_predictions(test_data, test_target, prediction, "outputs/demo_results.png")
    
    # Show model architecture
    print("\n🏗️  Model Architecture Summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print("\n✅ Demo completed! Check 'outputs/demo_results.png' for visualization")

def interactive_demo():
    """Interactive demo with multiple samples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainTumorSegmentationModel().to(device)
    
    try:
        model.load_state_dict(torch.load('checkpoints/final_model.pth', map_location=device))
    except:
        pass
    
    # Generate multiple samples
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for sample_idx in range(num_samples):
        test_data, test_target = create_synthetic_brain_data(1)
        test_data, test_target = test_data.to(device), test_target.to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model(test_data)
        
        # Take middle slice
        slice_idx = test_data.shape[2] // 2
        
        # Input (FLAIR modality)
        axes[sample_idx, 0].imshow(test_data[0, 2, slice_idx].cpu().numpy(), cmap='gray')
        axes[sample_idx, 0].set_title(f'Sample {sample_idx+1} - Input')
        axes[sample_idx, 0].axis('off')
        
        # Ground Truth
        gt_combined = torch.argmax(test_target[0, :, slice_idx], dim=0).cpu().numpy()
        axes[sample_idx, 1].imshow(gt_combined, cmap='jet')
        axes[sample_idx, 1].set_title('Ground Truth')
        axes[sample_idx, 1].axis('off')
        
        # Prediction
        pred_combined = torch.argmax(prediction[0, :, slice_idx], dim=0).cpu().numpy()
        axes[sample_idx, 2].imshow(pred_combined, cmap='jet')
        axes[sample_idx, 2].set_title('Prediction')
        axes[sample_idx, 2].axis('off')
        
        # Difference
        diff = np.abs(gt_combined - pred_combined)
        axes[sample_idx, 3].imshow(diff, cmap='hot')
        axes[sample_idx, 3].set_title('Difference')
        axes[sample_idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/interactive_demo.png', dpi=150, bbox_inches='tight')
    print("✅ Interactive demo saved as 'outputs/interactive_demo.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Demo')
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo with multiple samples')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        run_demo()