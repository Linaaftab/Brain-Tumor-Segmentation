# quick_start.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def quick_test():
    """Quick test to verify everything works"""
    print("🚀 Quick Start Test...")
    
    # Test basic imports
    from model import BrainTumorSegmentationModel
    from utils import create_synthetic_brain_data
    
    # Create model and data
    model = BrainTumorSegmentationModel()
    data, target = create_synthetic_brain_data(batch_size=1, depth=32, height=64, width=64)
    
    # Test forward pass
    with torch.no_grad():
        output = model(data)
    
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    
    # Quick visualization - use smaller slice index for smaller output
    slice_idx_input = 16  # For input (depth=32)
    slice_idx_output = 8  # For output (depth=16)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(data[0, 2, slice_idx_input].cpu().numpy(), cmap='gray')  # FLAIR modality
    plt.title('Input MRI (FLAIR)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    gt_combined = torch.argmax(target[0, :, slice_idx_input], dim=0).cpu().numpy()
    plt.imshow(gt_combined, cmap='jet')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    pred_combined = torch.argmax(output[0, :, slice_idx_output], dim=0).cpu().numpy()
    plt.imshow(pred_combined, cmap='jet')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/quick_test.png', dpi=150, bbox_inches='tight')
    
    print("✅ Quick test completed!")
    print(f"   Visualization saved: outputs/quick_test.png")
    
    return True

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    quick_test()