# test_environment.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def test_environment():
    print("🧪 Testing Environment Setup...")
    print("=" * 50)
    
    # Test PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Test other packages
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
    
    # Test folder structure
    folders = ['data', 'outputs', 'logs', 'checkpoints']
    for folder in folders:
        if os.path.exists(folder):
            print(f"✅ Folder '{folder}' exists")
        else:
            print(f"⚠️  Folder '{folder}' missing - creating now")
            os.makedirs(folder, exist_ok=True)
    
    # Create a simple tensor to test
    x = torch.randn(2, 4, 32, 32, 32)
    print(f"✅ Tensor creation works: {x.shape}")
    
    # Test model import
    try:
        from model import BrainTumorSegmentationModel
        model = BrainTumorSegmentationModel()
        print("✅ Model imported successfully")
        print(f"✅ Model parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Test forward pass
        test_input = torch.randn(1, 4, 32, 64, 64)
        output = model(test_input)
        print(f"✅ Forward pass works - Input: {test_input.shape}, Output: {output.shape}")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
    
    # Test utils import
    try:
        from utils import create_synthetic_brain_data
        data, target = create_synthetic_brain_data()
        print(f"✅ Utils imported successfully - Synthetic data: {data.shape}")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Environment setup completed successfully!")
    print("\n📝 Next steps:")
    print("   1. Run: python train.py --epochs 5 (to train the model)")
    print("   2. Run: python demo.py (to see results)")
    print("   3. Run: python demo.py --interactive (for multiple samples)")
    
    return True

if __name__ == "__main__":
    test_environment()