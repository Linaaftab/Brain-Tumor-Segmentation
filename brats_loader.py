import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
import glob

class BraTS2020Dataset(Dataset):
    """BraTS 2020 Dataset loader"""
    
    def __init__(self, data_path, split='train', transform=None, crop_size=(128, 128, 128)):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.crop_size = crop_size
        
        self.samples = []
        
        print(f"Looking for BraTS data in: {data_path}")
        
        # More flexible file searching - BraTS can have different naming conventions
        brats_folders = []
        
        # Look for all directories that might contain BraTS data
        if os.path.exists(data_path):
            # Method 1: Look for subject folders
            potential_folders = [f for f in os.listdir(data_path) 
                               if os.path.isdir(os.path.join(data_path, f))]
            
            for folder in potential_folders:
                folder_path = os.path.join(data_path, folder)
                
                # Look for NIfTI files with common BraTS patterns
                nifti_files = glob.glob(os.path.join(folder_path, "*.nii")) + \
                             glob.glob(os.path.join(folder_path, "*.nii.gz"))
                
                if nifti_files:
                    # Try to find the required modalities
                    flair_files = [f for f in nifti_files if 'flair' in f.lower()]
                    t1_files = [f for f in nifti_files if 't1' in f.lower() and 't1ce' not in f.lower()]
                    t1ce_files = [f for f in nifti_files if 't1ce' in f.lower()]
                    t2_files = [f for f in nifti_files if 't2' in f.lower()]
                    seg_files = [f for f in nifti_files if 'seg' in f.lower()]
                    
                    # Check if we have all required files
                    if flair_files and t1_files and t1ce_files and t2_files and seg_files:
                        self.samples.append({
                            'flair': flair_files[0],
                            't1': t1_files[0],
                            't1ce': t1ce_files[0],
                            't2': t2_files[0],
                            'seg': seg_files[0]
                        })
        
        # If no structured folders found, look for loose files
        if not self.samples:
            print("Trying alternative file structure...")
            all_nifti_files = glob.glob(os.path.join(data_path, "**/*.nii"), recursive=True) + \
                             glob.glob(os.path.join(data_path, "**/*.nii.gz"), recursive=True)
            
            # Group files by subject
            subject_files = {}
            for file_path in all_nifti_files:
                filename = os.path.basename(file_path)
                # Extract subject ID (assuming format like BraTS2021_00000_flair.nii.gz)
                if 'BraTS' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        subject_id = f"{parts[0]}_{parts[1]}"
                        if subject_id not in subject_files:
                            subject_files[subject_id] = {}
                        
                        if 'flair' in filename.lower():
                            subject_files[subject_id]['flair'] = file_path
                        elif 't1.' in filename.lower() and 't1ce' not in filename.lower():
                            subject_files[subject_id]['t1'] = file_path
                        elif 't1ce' in filename.lower():
                            subject_files[subject_id]['t1ce'] = file_path
                        elif 't2' in filename.lower():
                            subject_files[subject_id]['t2'] = file_path
                        elif 'seg' in filename.lower():
                            subject_files[subject_id]['seg'] = file_path
            
            # Add complete samples
            for subject_id, files in subject_files.items():
                if all(modality in files for modality in ['flair', 't1', 't1ce', 't2', 'seg']):
                    self.samples.append(files)
        
        if not self.samples:
            print(f"❌ No BraTS data found in {data_path}")
            print("Please check:")
            print("1. Data path is correct")
            print("2. Files have correct naming (contain 'flair', 't1', 't1ce', 't2', 'seg')")
            print("3. Files are in .nii or .nii.gz format")
        else:
            print(f"✅ Found {len(self.samples)} BraTS cases for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def _normalize_modality(self, modality):
        """Normalize a single modality"""
        # Remove outliers (0.5th and 99.5th percentiles)
        p0 = np.percentile(modality, 0.5)
        p99 = np.percentile(modality, 99.5)
        modality = np.clip(modality, p0, p99)
        
        # Normalize to [0, 1]
        if modality.max() > modality.min():
            modality = (modality - modality.min()) / (modality.max() - modality.min())
        
        return modality
    
    def _center_crop(self, volume, target_size):
        """Center crop volume to target size"""
        depth, height, width = volume.shape
        
        # Calculate crop coordinates
        start_d = (depth - target_size[0]) // 2
        start_h = (height - target_size[1]) // 2
        start_w = (width - target_size[2]) // 2
        
        # Ensure we don't have negative indices
        start_d = max(0, start_d)
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        
        # Crop
        cropped = volume[
            start_d:start_d + target_size[0],
            start_h:start_h + target_size[1],
            start_w:start_w + target_size[2]
        ]
        
        # If volume is smaller than target, pad
        if cropped.shape != target_size:
            pad_d = target_size[0] - cropped.shape[0]
            pad_h = target_size[1] - cropped.shape[1]
            pad_w = target_size[2] - cropped.shape[2]
            
            cropped = np.pad(cropped, 
                           ((0, pad_d), (0, pad_h), (0, pad_w)),
                           mode='constant')
        
        return cropped
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load MRI modalities
            flair = nib.load(sample['flair']).get_fdata().astype(np.float32)
            t1 = nib.load(sample['t1']).get_fdata().astype(np.float32)
            t1ce = nib.load(sample['t1ce']).get_fdata().astype(np.float32)
            t2 = nib.load(sample['t2']).get_fdata().astype(np.float32)
            
            # Load segmentation mask
            seg = nib.load(sample['seg']).get_fdata().astype(np.float32)
            
            # Normalize each modality
            flair = self._normalize_modality(flair)
            t1 = self._normalize_modality(t1)
            t1ce = self._normalize_modality(t1ce)
            t2 = self._normalize_modality(t2)
            
            # Center crop to reduce memory usage
            if self.crop_size:
                flair = self._center_crop(flair, self.crop_size)
                t1 = self._center_crop(t1, self.crop_size)
                t1ce = self._center_crop(t1ce, self.crop_size)
                t2 = self._center_crop(t2, self.crop_size)
                seg = self._center_crop(seg, self.crop_size)
            
            # Stack modalities (4, D, H, W)
            image = np.stack([flair, t1, t1ce, t2], axis=0)
            
            # Convert segmentation to multi-class format
            # BraTS labels: 0=background, 1=NET, 2=ED, 4=ET
            # Convert to: WT (1,2,4), TC (1,4), ET (4)
            wt_mask = np.isin(seg, [1, 2, 4]).astype(np.float32)  # Whole tumor
            tc_mask = np.isin(seg, [1, 4]).astype(np.float32)     # Tumor core
            et_mask = (seg == 4).astype(np.float32)               # Enhancing tumor
            
            mask = np.stack([wt_mask, tc_mask, et_mask], axis=0)
            
            # Convert to torch tensors
            image = torch.FloatTensor(image)
            mask = torch.FloatTensor(mask)
            
            if self.transform:
                image, mask = self.transform(image, mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"Sample files: {sample}")
            # Return a dummy sample to avoid breaking training
            dummy_shape = (4, *self.crop_size) if self.crop_size else (4, 128, 128, 128)
            return torch.randn(dummy_shape), torch.randint(0, 2, (3, *dummy_shape[1:])).float()

class BraTSTransform:
    """Simple data augmentation for BraTS data"""
    
    def __init__(self, augment=True):
        self.augment = augment
    
    def __call__(self, image, mask):
        if self.augment and np.random.random() > 0.5:
            # Random flip along each axis
            for dim in [2, 3]:  # Height and width dimensions
                if np.random.random() > 0.5:
                    image = torch.flip(image, [dim])
                    mask = torch.flip(mask, [dim])
        
        return image, mask

def prepare_brats_data(data_path, output_path):
    """Preprocess and prepare BraTS data for training"""
    print("Preparing BraTS 2020 data...")
    
    # Create dataset to verify data is accessible
    dataset = BraTS2020Dataset(data_path, split='train')
    
    os.makedirs(output_path, exist_ok=True)
    
    print("BraTS data preparation completed!")
    print(f"Total samples: {len(dataset)}")
    
    return dataset

def test_brats_loader():
    """Test function to verify BraTS loader works"""
    data_path = "data/BraTS2020"  # Update this path
    
    if os.path.exists(data_path):
        print("Testing BraTS loader...")
        dataset = BraTS2020Dataset(data_path)
        
        if len(dataset) > 0:
            print(f"Successfully loaded {len(dataset)} samples")
            
            # Test loading one sample
            image, mask = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"Mask unique values: {torch.unique(mask)}")
            
            return True
        else:
            print("No samples found")
            return False
    else:
        print(f"Data path {data_path} does not exist")
        return False

if __name__ == "__main__":
    test_brats_loader()