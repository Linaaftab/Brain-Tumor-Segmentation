# check_structure.py
import os

brats_path = "data/BraTS2020_TrainingData"

def check_structure(path):
    print(f"Checking: {path}")
    
    if not os.path.exists(path):
        print("❌ Path doesn't exist")
        return
    
    items = os.listdir(path)
    print(f"Found {len(items)} items")
    
    # Check first 3 patient folders
    for i, item in enumerate(items[:3]):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"\n📂 Patient folder {i+1}: {item}")
            files = os.listdir(item_path)
            print(f"   Files: {files}")
            
            # Check for required modalities
            has_flair = any('flair' in f.lower() for f in files)
            has_t1 = any('t1' in f.lower() and 't1ce' not in f.lower() for f in files)
            has_t1ce = any('t1ce' in f.lower() for f in files)
            has_t2 = any('t2' in f.lower() for f in files)
            has_seg = any('seg' in f.lower() for f in files)
            
            print(f"   Modalities found:")
            print(f"     - FLAIR: {'✅' if has_flair else '❌'}")
            print(f"     - T1: {'✅' if has_t1 else '❌'}")
            print(f"     - T1ce: {'✅' if has_t1ce else '❌'}")
            print(f"     - T2: {'✅' if has_t2 else '❌'}")
            print(f"     - SEG: {'✅' if has_seg else '❌'}")

check_structure(brats_path)