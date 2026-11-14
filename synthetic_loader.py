# synthetic_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np

class SyntheticBrainDataset(Dataset):
    """
    Dataset loader for pre-generated synthetic data
    """
    def __init__(self, split='train', data_dir='data/synthetic', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.samples = []
        
        # Load all sample files
        for file in os.listdir(self.data_dir):
            if file.endswith('.pkl'):
                self.samples.append(file)
        
        print(f"📁 Loaded {len(self.samples)} synthetic samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)
        
        data = sample['data']
        target = sample['target']
        
        # Apply transforms if any
        if self.transform:
            data, target = self.transform(data, target)
        
        return data, target
    
    def get_sample_info(self, idx):
        """Get metadata for a sample"""
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)
        return {
            'patient_id': sample['patient_id'],
            'modalities': sample['modalities'],
            'tumor_classes': sample['tumor_classes']
        }

def load_synthetic_dataset_info(data_dir='data/synthetic'):
    """Load dataset information"""
    info_path = os.path.join(data_dir, 'dataset_info.pkl')
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    return info

# Data augmentation transforms
class RandomAugmentation:
    def __init__(self):
        pass
    
    def __call__(self, data, seg):
        # Random flip
        if np.random.random() > 0.5:
            data = torch.flip(data, dims=[2])  # Flip height dimension
            seg = torch.flip(seg, dims=[2])
        
        if np.random.random() > 0.5:
            data = torch.flip(data, dims=[3])  # Flip width dimension
            seg = torch.flip(seg, dims=[3])
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)  # 1, 2, or 3 rotations of 90 degrees
            data = torch.rot90(data, k, dims=[2, 3])
            seg = torch.rot90(seg, k, dims=[2, 3])
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            data = data * factor
        
        return data, seg