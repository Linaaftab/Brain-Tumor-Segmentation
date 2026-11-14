# test_model.py
import torch
from model import BrainTumorSegmentationModel

print('Testing BraTS model loading...')
model = BrainTumorSegmentationModel()

# Try to load the BraTS final model
try:
    model.load_state_dict(torch.load('checkpoints/brats_final_model.pth', map_location='cpu'))
    print('✅ brats_final_model.pth loaded successfully!')
    
    # Test with a sample input
    dummy_input = torch.randn(1, 4, 32, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
        print(f'✅ Model works! Input: {dummy_input.shape}, Output: {output.shape}')
        
except Exception as e:
    print(f'❌ Error loading model: {e}')
    # Try the epoch 10 model instead
    try:
        model.load_state_dict(torch.load('checkpoints/brats_model_epoch_10.pth', map_location='cpu'))
        print('✅ brats_model_epoch_10.pth loaded successfully!')
    except Exception as e2:
        print(f'❌ Error loading epoch 10 model: {e2}')