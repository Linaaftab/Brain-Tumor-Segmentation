# generate_synthetic_data.py
import numpy as np
import os
from tqdm import tqdm

def generate_synthetic_brain_data(num_samples=100, depth=32, height=64, width=64):
    """Generate synthetic brain tumor data matching your model's expected format"""
    print("Generating synthetic brain tumor data...")
    
    # Create directories
    os.makedirs('data/synthetic/train/images', exist_ok=True)
    os.makedirs('data/synthetic/train/masks', exist_ok=True)
    os.makedirs('data/synthetic/val/images', exist_ok=True)
    os.makedirs('data/synthetic/val/masks', exist_ok=True)
    
    # Split into train/validation
    train_samples = int(num_samples * 0.8)
    val_samples = num_samples - train_samples
    
    print(f"Generating {train_samples} training samples and {val_samples} validation samples...")
    
    # Generate training data
    for i in tqdm(range(train_samples), desc="Training data"):
        # Generate synthetic MRI-like data (4 channels)
        image = np.random.randn(4, depth, height, width).astype(np.float32)
        # Generate synthetic tumor masks (3 channels for WT, TC, ET)
        mask = np.random.randint(0, 2, (3, depth, height, width)).astype(np.float32)
        
        # Save as numpy arrays
        np.save(f'data/synthetic/train/images/sample_{i:04d}.npy', image)
        np.save(f'data/synthetic/train/masks/sample_{i:04d}.npy', mask)
    
    # Generate validation data
    for i in tqdm(range(val_samples), desc="Validation data"):
        image = np.random.randn(4, depth, height, width).astype(np.float32)
        mask = np.random.randint(0, 2, (3, depth, height, width)).astype(np.float32)
        
        np.save(f'data/synthetic/val/images/sample_{i:04d}.npy', image)
        np.save(f'data/synthetic/val/masks/sample_{i:04d}.npy', mask)
    
    
    print("✅ Synthetic data generation completed!")
    print(f"📁 Data saved in:")
    print(f"   - data/synthetic/train/")
    print(f"   - data/synthetic/val/")

if __name__ == "__main__":
    generate_synthetic_brain_data(num_samples=50)  # Generate 50 samples for testing