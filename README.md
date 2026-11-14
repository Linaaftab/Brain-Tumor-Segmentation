# Brain Tumor Segmentation Using 3D U-Net



A deep learning project for 3D brain tumor segmentation using a 3D U-Net-style architecture built with PyTorch. The project supports synthetic brain MRI data and the BraTS 2020 dataset.



## Project Overview



This project explores automated brain tumor segmentation from multi-modal MRI data. It uses a 3D convolutional neural network to predict three tumor regions:



- **WT:** Whole Tumor

- **TC:** Tumor Core

- **ET:** Enhancing Tumor



The model accepts four MRI input channels and produces three segmentation output channels.



**Note:** This is an educational and research project, not a clinically validated diagnostic system. Predictions must not be used for medical diagnosis or treatment decisions.



## Model Architecture



The project uses a 3D U-Net-style encoder-decoder architecture featuring:



- 3D convolutional layers

- Batch normalization

- ReLU activation

- Transposed convolution for upsampling

- Encoder-decoder skip connections

- Three-channel segmentation output



The default model configuration uses:

- Input channels: 4

- Output channels: 3

- Base channels: 16



## Features



- 3D brain tumor segmentation

- Synthetic MRI data generation

- BraTS 2020 dataset support

- Training and validation monitoring

- Dice score calculation for WT, TC and ET

- Model checkpoint saving and loading

- Desktop graphical user interface

- Custom image upload

- Synthetic MRI visualization

- Model comparison interface

- Training progress visualization



## Technologies Used



- Python

- PyTorch

- MONAI

- NumPy

- NiBabel

- OpenCV

- Matplotlib

- Scikit-learn

- Tkinter



## Project Structure



```text

Brain-Tumor-Segmentation/

│

├── checkpoints/

│   ├── brats_final_model.pth

│   ├── final_model.pth

│   └── synthetic_old_final_model.pth

│

├── outputs/

│   ├── brats_training_progress.png

│   ├── demo_results.png

│   ├── quick_test.png

│   └── training_progress.png

│

├── test_images/

│

├── model.py

├── train.py

├── utils.py

├── brats_loader.py

├── synthetic_loader.py

├── generate_synthetic_data.py

├── gui_app_new.py

├── demo.py

├── quick_start.py

├── test_model.py

├── test_environment.py

├── requirements.txt

└── README.md

```



## Installation



### 1. Clone the repository



```bash

git clone https://github.com/Linaaftab/Brain-Tumor-Segmentation.git

cd Brain-Tumor-Segmentation

```



### 2. Create a virtual environment



```bash

python -m venv brain_tumor_env

```



Activate it on Windows:



```powershell

.\\brain_tumor_env\\Scripts\\Activate.ps1

```



### 3. Install dependencies



```bash

pip install -r requirements.txt

```



A compatible PyTorch installation is required. For GPU acceleration, install a PyTorch version compatible with your CUDA environment.



## Dataset



The project supports two types of data:



### 1. Synthetic Dataset



Synthetic MRI-like data can be generated for testing and experimentation.



Run:



```bash

python generate_synthetic_data.py

```



The generator creates synthetic four-channel images and three-channel tumor masks.



**Important:** These are randomly generated samples, not real patient MRI scans. Synthetic results should not be interpreted as evidence of clinical segmentation performance.



### 2. BraTS 2020 Dataset



The project also supports the BraTS 2020 brain tumor segmentation dataset.



The loader expects five medical image files for each patient:



- FLAIR

- T1

- T1ce

- T2

- Segmentation mask



Supported medical image formats include `.nii` and `.nii.gz`.



Download and prepare the dataset separately according to its official access and usage requirements. The dataset is not included in this repository.



The default dataset path is:



```text

data/BraTS2020/

```



## Model Training



The training script supports three dataset modes:



- `synthetic_saved` — pre-saved synthetic dataset (default)

- `synthetic_old` — synthetic data generated during training

- `brats` — BraTS 2020 dataset



### Train using the default dataset



```bash

python train.py

```



### Train using BraTS 2020



```bash

python train.py --dataset brats --brats_path data/BraTS2020

```



### Default Training Configuration



| Parameter | Value |

|---|---:|

| Epochs | 50 |

| Batch size | 2 |

| Learning rate | 0.001 |

| Weight decay | 0.00001 |

| Input channels | 4 |

| Output channels | 3 |

| Base channels | 16 |



Training checkpoints are saved in the `checkpoints/` directory, while training visualizations are saved in `outputs/`.



## Training Visualization



The training visualization contains two plots:



1\. **Training Loss:** Displays the change in training loss across epochs.

2\. **Dice Scores:** Displays Dice scores for the three segmentation output channels.



The graphs are generated automatically during training.



Example output:



```text

outputs/

└── brats_training_progress.png

```



## Running the Demo



Run the standard demo:



```bash

python demo.py

```



Run the interactive demo:



```bash

python demo.py --interactive

```



The demo generates synthetic MRI-like input data, performs model inference and calculates Dice scores for the three tumor regions.



## Graphical User Interface



Launch the desktop application:



```bash

python gui_app_new.py

```



The graphical interface includes:



- Synthetic data selection

- BraTS 2020 dataset selection

- Custom image upload

- Tumor detection

- Model comparison

- Synthetic MRI generation

- BraTS dataset loading

- Model checkpoint loading



Supported custom image formats include JPG, JPEG, PNG, BMP and TIFF.



## Evaluation Metrics



The project calculates Dice scores for three tumor regions:



- Whole Tumor (WT)

- Tumor Core (TC)

- Enhancing Tumor (ET)



Training and validation loss are also monitored during the training process.



No clinical performance claims are made from synthetic test data.



## Outputs



The project generates several visual outputs, including:



- Training progress graphs

- Dice score plots

- Segmentation demonstrations

- Model architecture figures

- Test result visualizations



## Future Improvements



Potential future improvements include:



- More extensive BraTS 2020 evaluation

- Improved data augmentation

- Hyperparameter optimization

- Model performance evaluation on independent test data

- Enhanced segmentation visualization

- Further GUI improvements



## Disclaimer



This project is developed for educational and research purposes. It is not intended for clinical diagnosis, medical treatment, or replacement of professional medical judgment.



## Author



**Aftab Aleena**



GitHub: \[Linaaftab](https://github.com/Linaaftab)



Project Repository: \[Brain Tumor Segmentation](https://github.com/Linaaftab/Brain-Tumor-Segmentation)

