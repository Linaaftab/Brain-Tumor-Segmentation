# live_webcam_demo.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from model import BrainTumorSegmentationModel
from utils import create_synthetic_brain_data
import torch.nn.functional as F

class LiveBrainTumorDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Load model
        self.model = BrainTumorSegmentationModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load('checkpoints/final_model.pth', map_location=self.device))
            print("✅ Model loaded successfully!")
        except:
            print("❌ No trained model found. Using random weights for demo.")
        
        self.model.eval()
        
        # Setup visualization
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        plt.suptitle('LIVE Brain Tumor Detection - GPU Accelerated', fontsize=16, fontweight='bold')
        
    def simulate_mri_slice(self):
        """Simulate getting a new MRI slice with safe dimensions"""
        # Use dimensions that work with the random generator
        data, target = create_synthetic_brain_data(
            batch_size=1, 
            depth=32,    # Safe minimum depth
            height=128,  # Reasonable size
            width=128
        )
        return data.to(self.device), target.to(self.device)
    
    def process_frame(self, data):
        """Process one frame with GPU acceleration"""
        start_time = time.time()
        
        with torch.no_grad():
            prediction = self.model(data)
        
        processing_time = time.time() - start_time
        return prediction, processing_time
    
    def visualize_results(self, data, target, prediction, processing_time, frame_count):
        """Update live visualization"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Take middle slice for display
        slice_idx = data.shape[2] // 2
        
        # Input modalities
        modalities = ['T1', 'T2', 'FLAIR', 'T1ce']
        for i in range(min(4, data.shape[1])):
            img = data[0, i, slice_idx].cpu().numpy()
            self.axes[0, i].imshow(img, cmap='gray')
            self.axes[0, i].set_title(f'Input: {modalities[i]}')
            self.axes[0, i].axis('off')
        
        # Ground Truth (if available)
        gt_combined = torch.argmax(target[0, :, slice_idx], dim=0).cpu().numpy()
        self.axes[1, 0].imshow(gt_combined, cmap='jet')
        self.axes[1, 0].set_title('Ground Truth')
        self.axes[1, 0].axis('off')
        
        # Prediction
        prediction_resized = F.interpolate(
            prediction, size=target.shape[-3:], mode='trilinear', align_corners=True
        )
        pred_combined = torch.argmax(prediction_resized[0, :, slice_idx], dim=0).cpu().numpy()
        self.axes[1, 1].imshow(pred_combined, cmap='jet')
        self.axes[1, 1].set_title('AI Prediction')
        self.axes[1, 1].axis('off')
        
        # Overlay (Input + Prediction)
        input_img = data[0, 2, slice_idx].cpu().numpy()  # FLAIR channel
        self.axes[1, 2].imshow(input_img, cmap='gray')
        self.axes[1, 2].imshow(pred_combined, cmap='jet', alpha=0.5)
        self.axes[1, 2].set_title('Overlay: MRI + Detection')
        self.axes[1, 2].axis('off')
        
        # Metrics and Info
        from utils import calculate_metrics
        dice_scores = calculate_metrics(prediction, target)
        
        info_text = f"""
Frame: {frame_count}
Processing: {processing_time*1000:.1f}ms
Device: {self.device}

DICE SCORES:
Whole Tumor: {dice_scores[0]:.4f}
Tumor Core: {dice_scores[1]:.4f}
Enhancing: {dice_scores[2]:.4f}
Average: {np.mean(dice_scores):.4f}

STATUS: {'✅ DETECTED' if np.mean(dice_scores) > 0.1 else '⚠️ SCANNING'}
"""
        
        self.axes[1, 3].text(0.05, 0.9, info_text, fontsize=10, fontfamily='monospace', 
                            verticalalignment='top', transform=self.axes[1, 3].transAxes)
        self.axes[1, 3].axis('off')
        
        plt.draw()
        plt.pause(0.01)
    
    def run_live_detection(self, duration_seconds=30):
        """Run live detection for specified duration"""
        print(f"🎥 Starting LIVE detection for {duration_seconds} seconds...")
        print("Press Ctrl+C in the terminal to stop early")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                frame_count += 1
                
                # Simulate getting new MRI data
                data, target = self.simulate_mri_slice()
                
                # Process with GPU
                prediction, processing_time = self.process_frame(data)
                
                # Update display
                self.visualize_results(data, target, prediction, processing_time, frame_count)
                
                # Simulate real-time frame rate
                time.sleep(1.0)  # 1 FPS for stable demo
                
        except KeyboardInterrupt:
            print("\n🛑 Live detection stopped by user")
        
        print(f"🎯 Live detection completed! Processed {frame_count} frames")
        plt.ioff()
        plt.show()

def main():
    detector = LiveBrainTumorDetector()
    detector.run_live_detection(duration_seconds=20)  # 20-second demo

if __name__ == "__main__":
    main()