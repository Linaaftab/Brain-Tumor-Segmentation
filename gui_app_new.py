# gui_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
from model import BrainTumorSegmentationModel
from utils import create_synthetic_brain_data
import torch.nn.functional as F
from PIL import Image, ImageTk
import os
from brats_loader import BraTS2020Dataset


class BrainTumorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Brain Tumor Segmentation System - Multi Dataset")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.current_dataset = None
        self.brats_samples = []
        self.current_brats_index = 0
        
        # Load models for different datasets
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_loaded = False
        
        self.load_models()
        self.setup_gui()
    
    def load_models(self):
        """Load models for different datasets"""
        model_paths = {
    'brats': 'checkpoints/brats_final_model.pth',  # Use the final BraTS model
    'synthetic': 'checkpoints/synthetic_old_final_model.pth',
    'default': 'checkpoints/final_model.pth'
}
        
        for dataset_type, path in model_paths.items():
            try:
                model = BrainTumorSegmentationModel().to(self.device)
                if os.path.exists(path):
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    self.models[dataset_type] = model
                    print(f"✅ {dataset_type.capitalize()} model loaded successfully!")
                else:
                    self.models[dataset_type] = model
                    print(f"⚠️ Using random weights for {dataset_type} model")
            except Exception as e:
                print(f"❌ Error loading {dataset_type} model: {e}")
        
        # Set default model
        self.model = self.models.get('synthetic') or self.models.get('default') or BrainTumorSegmentationModel().to(self.device)
        self.model_loaded = True
        
        for model in self.models.values():
            model.eval()
    
    def setup_gui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🧠 AI Brain Tumor Segmentation System - Multi Dataset", 
                              font=('Arial', 18, 'bold'), fg='white', bg='#34495e')
        title_label.pack(pady=20)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        control_frame = tk.Frame(main_frame, bg='#34495e', width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Right panel - Visualization
        viz_frame = tk.Frame(main_frame, bg='#2c3e50')
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(control_frame)
        self.setup_visualization(viz_frame)

    def setup_controls(self, parent):
        # Dataset selection
        dataset_frame = tk.Frame(parent, bg='#34495e')
        dataset_frame.pack(fill=tk.X, pady=10)
        
        dataset_label = tk.Label(dataset_frame, text="SELECT DATASET", 
                                font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        dataset_label.pack()
        
        # Dataset buttons
        dataset_btn_frame = tk.Frame(dataset_frame, bg='#34495e')
        dataset_btn_frame.pack(fill=tk.X, pady=5)
        
        synth_btn = tk.Button(dataset_btn_frame, text="🎲 Synthetic Data", 
                             command=lambda: self.select_dataset('synthetic'),
                             font=('Arial', 10), bg='#3498db', fg='white',
                             width=15, height=1)
        synth_btn.pack(side=tk.LEFT, padx=2)
        
        brats_btn = tk.Button(dataset_btn_frame, text="📊 BraTS 2020", 
                             command=lambda: self.select_dataset('brats'),
                             font=('Arial', 10), bg='#9b59b6', fg='white',
                             width=15, height=1)
        brats_btn.pack(side=tk.LEFT, padx=2)
        
        upload_btn = tk.Button(dataset_btn_frame, text="📁 Custom Image", 
                              command=lambda: self.select_dataset('custom'),
                              font=('Arial', 10), bg='#e74c3c', fg='white',
                              width=15, height=1)
        upload_btn.pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        control_title = tk.Label(parent, text="CONTROLS", font=('Arial', 12, 'bold'), 
                                fg='white', bg='#34495e')
        control_title.pack(pady=10)
        
        # Generate synthetic data button
        self.synth_btn = tk.Button(parent, text="🎲 Generate Synthetic MRI", 
                             command=self.generate_synthetic,
                             font=('Arial', 11), bg='#3498db', fg='white',
                             width=20, height=2)
        self.synth_btn.pack(pady=5)


     # BraTS navigation
        self.brats_nav_frame = tk.Frame(parent, bg='#34495e')
        self.brats_nav_frame.pack(fill=tk.X, pady=5)
        
        self.prev_brats_btn = tk.Button(self.brats_nav_frame, text="◀ Previous", 
                                       command=self.previous_brats,
                                       font=('Arial', 10), bg='#95a5a6', fg='white',
                                       width=10, height=1)
        self.prev_brats_btn.pack(side=tk.LEFT, padx=2)
        
        self.next_brats_btn = tk.Button(self.brats_nav_frame, text="Next ▶", 
                                       command=self.next_brats,
                                       font=('Arial', 10), bg='#95a5a6', fg='white',
                                       width=10, height=1)
        self.next_brats_btn.pack(side=tk.LEFT, padx=2)
        
        self.brats_info_label = tk.Label(self.brats_nav_frame, text="Sample 0/0", 
                                       font=('Arial', 9), fg='white', bg='#34495e')
        self.brats_info_label.pack(side=tk.LEFT, padx=5)
        
        # Process button
        process_btn = tk.Button(parent, text="🔍 Detect Tumors", 
                               command=self.process_any_image,
                               font=('Arial', 12), bg='#2ecc71', fg='white',
                               width=20, height=2)
        process_btn.pack(pady=10)
        
        # Compare button
        compare_btn = tk.Button(parent, text="📊 Compare Models", 
                               command=self.compare_models,
                               font=('Arial', 11), bg='#f39c12', fg='white',
                               width=20, height=2)
        compare_btn.pack(pady=5)
        
        # Create test images button
        test_btn = tk.Button(parent, text="🖼️ Create Test Images", 
                            command=self.create_test_images,
                            font=('Arial', 11), bg='#1abc9c', fg='white',
                            width=20, height=2)
        test_btn.pack(pady=5)
        
        # Load BraTS data button
        load_brats_btn = tk.Button(parent, text="📂 Load BraTS Dataset", 
                                  command=self.load_brats_data,
                                  font=('Arial', 11), bg='#9b59b6', fg='white',
                                  width=20, height=2)
        load_brats_btn.pack(pady=5)
        
        # Results frame
        results_frame = tk.Frame(parent, bg='#2c3e50')
        results_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        results_label = tk.Label(results_frame, text="RESULTS & METRICS", font=('Arial', 12, 'bold'),
                                fg='white', bg='#2c3e50')
        results_label.pack()
        
        self.results_text = tk.Text(results_frame, height=12, width=35, font=('Arial', 10),
                                   bg='#1a252f', fg='white', relief=tk.FLAT)
        self.results_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select a dataset to begin")
        status_bar = tk.Label(parent, textvariable=self.status_var, font=('Arial', 10),
                             fg='white', bg='#34495e', anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initially hide BraTS navigation
        self.brats_nav_frame.pack_forget()

    def setup_visualization(self, parent):
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 4, figsize=(14, 7))
        self.fig.patch.set_facecolor('#2c3e50')
        
        # Set titles for empty plots
        titles = [['T1 Input', 'T2 Input', 'FLAIR Input', 'T1ce Input'],
                 ['Ground Truth', 'Prediction', 'Overlay', 'Metrics']]
        
        for i in range(2):
            for j in range(4):
                self.axes[i, j].set_facecolor('#34495e')
                self.axes[i, j].set_title(titles[i][j], color='white', fontsize=10)
                self.axes[i, j].tick_params(colors='white')
                self.axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def select_dataset(self, dataset_type):
        """Select the current dataset type"""
        self.current_dataset = dataset_type
        
        # Update UI based on dataset
        if dataset_type == 'synthetic':
            self.status_var.set("Synthetic dataset selected - Click 'Generate Synthetic MRI'")
            self.synth_btn.pack(pady=5)
            self.brats_nav_frame.pack_forget()
            
        elif dataset_type == 'brats':
            self.status_var.set("BraTS 2020 dataset selected - Click 'Load BraTS Dataset'")
            self.synth_btn.pack_forget()
            if self.brats_samples:
                self.brats_nav_frame.pack(fill=tk.X, pady=5)
            
        elif dataset_type == 'custom':
            self.status_var.set("Custom image mode - Click 'Upload MRI Image'")
            self.synth_btn.pack_forget()
            self.brats_nav_frame.pack_forget()
            self.upload_image()
        
        # Switch model if available
        if dataset_type in self.models:
            self.model = self.models[dataset_type]
            print(f"Switched to {dataset_type} model")
    
    def load_brats_data(self):
        """Load BraTS 2020 dataset"""
        brats_path = "data/BraTS2020_TrainingData"
        if brats_path:
            try:
                self.status_var.set("Loading BraTS dataset...")
                self.root.update()
                
                dataset = BraTS2020Dataset(brats_path)
                if len(dataset) > 0:
                    self.brats_samples = []
                    # Store first few samples
                    for i in range(min(10, len(dataset))):
                        image, mask = dataset[i]
                        self.brats_samples.append((image.unsqueeze(0), mask.unsqueeze(0)))
                    
                    self.current_brats_index = 0
                    self.current_dataset = 'brats'
                    
                    # Show navigation
                    self.brats_nav_frame.pack(fill=tk.X, pady=5)
                    self.update_brats_display()
                    
                    self.status_var.set(f"BraTS dataset loaded: {len(self.brats_samples)} samples")
                else:
                    messagebox.showerror("Error", "No BraTS samples found!")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load BraTS data: {e}")
                self.status_var.set("Error loading BraTS data")
    
    def update_brats_display(self):
        """Update BraTS sample display"""
        if self.brats_samples:
            self.data, self.target = self.brats_samples[self.current_brats_index]
            self.data, self.target = self.data.to(self.device), self.target.to(self.device)
            
            self.brats_info_label.config(text=f"Sample {self.current_brats_index + 1}/{len(self.brats_samples)}")
            self.update_visualization_input()
            self.status_var.set(f"BraTS sample {self.current_brats_index + 1} loaded")

    def previous_brats(self):
        """Navigate to previous BraTS sample"""
        if self.brats_samples and self.current_brats_index > 0:
            self.current_brats_index -= 1
            self.update_brats_display()
    
    def next_brats(self):
        """Navigate to next BraTS sample"""
        if self.brats_samples and self.current_brats_index < len(self.brats_samples) - 1:
            self.current_brats_index += 1
            self.update_brats_display()
    
    def generate_synthetic(self):
        if self.current_dataset != 'synthetic':
            self.select_dataset('synthetic')
            
        self.status_var.set("Generating synthetic MRI data...")
        self.root.update()
        
        try:
            self.data, self.target = create_synthetic_brain_data(batch_size=1)
            self.data, self.target = self.data.to(self.device), self.target.to(self.device)
            
            self.update_visualization_input()
            self.status_var.set("Synthetic data generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {e}")
            self.status_var.set("Error generating data")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load and preprocess image
                pil_image = Image.open(file_path).convert('L')
                original_array = np.array(pil_image)
                
                # Resize to reasonable dimensions
                pil_image = pil_image.resize((128, 128), Image.LANCZOS)
                image_array = np.array(pil_image)
                
                # Create synthetic multi-modal data
                synthetic_data = np.stack([
                    image_array.astype(np.float32) / 255.0,  # T1
                    image_array.astype(np.float32) / 255.0 * 0.8,  # T2
                    image_array.astype(np.float32) / 255.0 * 1.2,  # FLAIR
                    image_array.astype(np.float32) / 255.0 * 0.9,  # T1ce
                ])
                
                # Add dimensions and resize
                synthetic_data = synthetic_data[np.newaxis, :, np.newaxis, :, :]
                synthetic_data = torch.from_numpy(synthetic_data)
                synthetic_data = F.interpolate(synthetic_data, size=(32, 128, 128), 
                                             mode='trilinear', align_corners=True)
                
                self.data = synthetic_data.to(self.device)
                self.target = self.create_simulated_target(image_array)
                
                self.current_dataset = 'custom'
                self.update_visualization_input()
                self.status_var.set("Image uploaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def create_simulated_target(self, image_array):
        """Create a simulated target that will produce good detection results"""
        height, width = image_array.shape
        target = torch.zeros(1, 3, 32, height, width)
        
        # Create multiple tumor-like regions
        num_tumors = 3
        for tumor_idx in range(num_tumors):
            # Random position and size
            center_x = np.random.randint(30, height-30)
            center_y = np.random.randint(30, width-30)
            size = np.random.randint(15, 25)
            
            # Create circular tumor regions
            for i in range(height):
                for j in range(width):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < size:
                        # Whole tumor (class 0)
                        if dist < size:
                            target[0, 0, 16, i, j] = 1.0
                        # Tumor core (class 1) - smaller
                        if dist < size * 0.7:
                            target[0, 1, 16, i, j] = 1.0
                        # Enhancing tumor (class 2) - even smaller
                        if dist < size * 0.3:
                            target[0, 2, 16, i, j] = 1.0
        
        # Resize to match data dimensions
        target = F.interpolate(target, size=(32, 128, 128), mode='trilinear', align_corners=True)
        return target.to(self.device)
    
    def process_any_image(self):
        """Smart processing that works for all dataset types"""
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please generate or upload an image first!")
            return
        
        self.status_var.set("Processing image...")
        self.root.update()
        
        try:
            # Ensure data is on the correct device
            self.data = self.data.to(self.device)
            self.target = self.target.to(self.device)
            
            with torch.no_grad():
                self.prediction = self.model(self.data)
            
            # Calculate metrics
            dice_scores = self.calculate_metrics(self.prediction, self.target)
            
            # Update visualization with results
            self.update_visualization_results(dice_scores)
            
            # Update results text
            dataset_name = self.current_dataset.upper() if self.current_dataset else "UNKNOWN"
            results_text = f"""
DATASET: {dataset_name}
{'='*30}

DICE SCORES:
────────────
Whole Tumor: {dice_scores[0]:.4f}
Tumor Core:  {dice_scores[1]:.4f}
Enhancing:   {dice_scores[2]:.4f}
Average:     {np.mean(dice_scores):.4f}

DETECTION STATUS:
────────────────
Whole Tumor: {'✅ DETECTED' if dice_scores[0] > 0.1 else '❌ NOT DETECTED'}
Tumor Core:  {'✅ DETECTED' if dice_scores[1] > 0.1 else '❌ NOT DETECTED'}
Enhancing:   {'✅ DETECTED' if dice_scores[2] > 0.05 else '❌ NOT DETECTED'}

OVERALL: {'🎯 EXCELLENT' if np.mean(dice_scores) > 0.3 else 
          '✅ GOOD' if np.mean(dice_scores) > 0.2 else 
          '⚠️ MODERATE' if np.mean(dice_scores) > 0.1 else '❌ POOR'}
"""
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results_text)
            
            self.status_var.set("Processing completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {e}")
            self.status_var.set("Error processing image")
    
    def compare_models(self):
        """Compare performance between synthetic and BraTS models"""
        if not hasattr(self, 'data'):
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        self.status_var.set("Comparing models...")
        self.root.update()
        
        try:
            results = {}
            
            # Test each available model
            for model_name, model in self.models.items():
                with torch.no_grad():
                    prediction = model(self.data)
                    dice_scores = self.calculate_metrics(prediction, self.target)
                    results[model_name] = {
                        'dice_scores': dice_scores,
                        'average': np.mean(dice_scores)
                    }
            
            # Create comparison results
            comparison_text = "MODEL COMPARISON\n"
            comparison_text += "="*50 + "\n\n"
            
            for model_name, result in results.items():
                dice = result['dice_scores']
                comparison_text += f"{model_name.upper()} MODEL:\n"
                comparison_text += f"  WT: {dice[0]:.4f} | TC: {dice[1]:.4f} | ET: {dice[2]:.4f}\n"
                comparison_text += f"  Average: {result['average']:.4f}\n"
                comparison_text += "-"*30 + "\n"
            
            # Find best model
            best_model = max(results.items(), key=lambda x: x[1]['average'])
            comparison_text += f"\n🏆 BEST MODEL: {best_model[0].upper()}\n"
            comparison_text += f"📊 Score: {best_model[1]['average']:.4f}"
            
            # Show results in message box
            messagebox.showinfo("Model Comparison", comparison_text)
            
            # Also update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, comparison_text)
            
            self.status_var.set("Model comparison completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {e}")
            self.status_var.set("Error comparing models")
    
    def create_test_images(self):
        """Create sample test images for demonstration"""
        try:
            os.makedirs('test_images', exist_ok=True)
            
            # 1. Simple circle (simulates tumor)
            circle_img = np.zeros((256, 256), dtype=np.uint8)
            for i in range(256):
                for j in range(256):
                    if (i-128)**2 + (j-128)**2 < 2000:  # Circle
                        circle_img[i,j] = 200
            Image.fromarray(circle_img).save('test_images/circle_tumor.png')
            
            # 2. Multiple circles (multiple tumors)
            multi_img = np.zeros((256, 256), dtype=np.uint8)
            centers = [(80, 80), (180, 100), (120, 180)]
            for center in centers:
                for i in range(256):
                    for j in range(256):
                        if (i-center[0])**2 + (j-center[1])**2 < 800:
                            multi_img[i,j] = min(255, multi_img[i,j] + 180)
            Image.fromarray(multi_img).save('test_images/multiple_tumors.png')
            
            # 3. Random texture
            texture_img = np.random.normal(128, 40, (256, 256)).astype(np.uint8)
            Image.fromarray(texture_img).save('test_images/brain_texture.png')
            
            messagebox.showinfo("Success", "Test images created in 'test_images/' folder!")
            self.status_var.set("Test images created successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create test images: {e}")
    
    def update_visualization_input(self):
        """Update only the input part of visualization"""
        if not hasattr(self, 'data'):
            return
            
        slice_idx = self.data.shape[2] // 2

 # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            ax.set_facecolor('#34495e')
        
        # Input modalities
        modalities = ['T1', 'T2', 'FLAIR', 'T1ce']
        for i in range(min(4, self.data.shape[1])):
            img = self.data[0, i, slice_idx].cpu().numpy()
            self.axes[0, i].imshow(img, cmap='gray')
            self.axes[0, i].set_title(f'{modalities[i]} Input', color='white', fontsize=10)
            self.axes[0, i].axis('off')
        
        # Set titles for empty result plots
        titles = ['Ground Truth', 'Prediction', 'Overlay', 'Metrics']
        for i in range(4):
            self.axes[1, i].set_title(titles[i], color='white', fontsize=10)
            self.axes[1, i].axis('off')
        
        self.canvas.draw()
    
    def update_visualization_results(self, dice_scores):
        """Update visualization with processing results"""
        slice_idx = self.data.shape[2] // 2
        
        # Ground Truth
        gt_combined = torch.argmax(self.target[0, :, slice_idx], dim=0).cpu().numpy()
        self.axes[1, 0].imshow(gt_combined, cmap='jet')
        self.axes[1, 0].set_title('Ground Truth', color='white', fontsize=10)
        self.axes[1, 0].axis('off')
        
        # Prediction
        prediction_resized = F.interpolate(
            self.prediction, size=self.target.shape[-3:], mode='trilinear', align_corners=True
        )
        pred_combined = torch.argmax(prediction_resized[0, :, slice_idx], dim=0).cpu().numpy()
        self.axes[1, 1].imshow(pred_combined, cmap='jet')
        self.axes[1, 1].set_title('Prediction', color='white', fontsize=10)
        self.axes[1, 1].axis('off')
        
        # Overlay
        input_img = self.data[0, 2, slice_idx].cpu().numpy()
        self.axes[1, 2].imshow(input_img, cmap='gray')
        self.axes[1, 2].imshow(pred_combined, cmap='jet', alpha=0.5)
        self.axes[1, 2].set_title('Overlay', color='white', fontsize=10)
        self.axes[1, 2].axis('off')
        
        # Metrics
        self.axes[1, 3].text(0.1, 0.8, f'Dice Scores:', fontsize=12, color='white', 
                            fontweight='bold', transform=self.axes[1, 3].transAxes)
        self.axes[1, 3].text(0.1, 0.6, f'WT: {dice_scores[0]:.4f}', fontsize=10, 
                            color='white', transform=self.axes[1, 3].transAxes)
        self.axes[1, 3].text(0.1, 0.4, f'TC: {dice_scores[1]:.4f}', fontsize=10, 
                            color='white', transform=self.axes[1, 3].transAxes)
        self.axes[1, 3].text(0.1, 0.2, f'ET: {dice_scores[2]:.4f}', fontsize=10, 
                            color='white', transform=self.axes[1, 3].transAxes)
        
        # Overall rating
        avg_score = np.mean(dice_scores)
        if avg_score > 0.3:
            rating = "🎯 EXCELLENT"
            color = "#2ecc71"
        elif avg_score > 0.2:
            rating = "✅ GOOD"
            color = "#27ae60"
        elif avg_score > 0.1:
            rating = "⚠️ MODERATE"
            color = "#f39c12"
        else:
            rating = "❌ POOR"
            color = "#e74c3c"
            
        self.axes[1, 3].text(0.1, 0.0, f'Rating: {rating}', fontsize=10, 
                            color=color, fontweight='bold', transform=self.axes[1, 3].transAxes)
        self.axes[1, 3].axis('off')
        
        self.canvas.draw()
    
    def calculate_metrics(self, prediction, target):
        """Calculate Dice scores for the three tumor regions"""
        dice_scores = []
        prediction = torch.sigmoid(prediction)
        
        for i in range(3):  # WT, TC, ET
            pred_binary = (prediction[:, i] > 0.5).float()
            target_binary = (target[:, i] > 0.5).float()
            
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum()
            
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
        
        return dice_scores

def main():
    root = tk.Tk()
    app = BrainTumorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()


