# test_my_brats.py
from brats_loader import BraTS2020Dataset
import torch
# Test with your exact path
brats_path = "data/BraTS2020_TrainingData"

print(f"Testing BraTS data at: {brats_path}")
try:
    dataset = BraTS2020Dataset(brats_path)
    print(f"✅ Success! Loaded {len(dataset)} BraTS samples")
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"   Image shape: {image.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"   Mask unique values: {torch.unique(mask)}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()