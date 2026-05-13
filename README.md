<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🖼️ Image Reconstruction Under Corruption</h1>

<p align="center">
  <b>A deep learning approach to reconstruct clean images from corrupted counterparts using a DnCNN-style U-Net architecture.</b>
</p>

<p align="center">
  <i>Trained on 48,000 image pairs • Achieves 76.3% MSE improvement over baseline • Boosted with Test-Time Augmentation</i>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Model Architecture](#model-architecture)
  - [Loss Function](#loss-function)
  - [Training Strategy](#training-strategy)
  - [Test-Time Augmentation](#test-time-augmentation)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

This project tackles the challenge of **image reconstruction under corruption** — learning to reconstruct clean images from their corrupted versions using deep learning. The solution employs a powerful **DnCNN-style U-Net** with residual blocks, trained on paired clean/corrupted images, and enhanced with **Test-Time Augmentation (TTA)** for improved inference.

> **Key Highlight:** The model achieves a **76.3% reduction in MSE** compared to the corrupted baseline, bringing the MSE from **4822.19 → 1085.95** (with TTA) on the validation set.

---

## 🧩 Problem Statement

| Attribute          | Detail                                          |
| :----------------- | :---------------------------------------------- |
| **Goal**           | Reconstruct clean images from corrupted inputs   |
| **Metric**         | Mean Squared Error (MSE) — lower is better       |
| **Image Size**     | 32 × 32 RGB                                     |
| **Training Data**  | 48,000 paired (corrupted ↔ clean) images         |
| **Test Data**      | 12,000 corrupted images (no ground truth)         |

Given a corrupted input image, the model must predict the pixel values of the original clean image. The evaluation metric is **MSE computed on the [0, 255] pixel scale**.

---

## 📦 Dataset

The dataset is sourced from a Kaggle competition and consists of:

```
Datasets/
├── train_clean/       # 48,000 clean PNG images (32×32 RGB)
├── train_corrupt/     # 48,000 corresponding corrupted PNG images
├── test_corrupt/      # 12,000 corrupted images for submission
├── train.csv          # Metadata: id, class, clean_filename, corrupt_filename
└── sample_submission.csv
```

Images span **10 classes** (e.g., dog, automobile, horse, airplane, etc.) and exhibit various corruption types including:
- 🎨 **Color saturation manipulation** (~58% of images)
- ⬛ **Black pixel occlusion** (~24% of images)
- 🌫️ **Blur and noise** artifacts

---

## 🏗️ Approach

### Exploratory Data Analysis

The notebook performs comprehensive EDA to understand the nature of corruptions:

1. **Visual Inspection** — Side-by-side comparison of clean, corrupted, and absolute difference images
2. **Corruption Profiling** — Statistical analysis of 500 sample images measuring:
   - Saturation percentage (pixels with saturation > 200)
   - Black pixel percentage
   - Laplacian variance (sharpness)
3. **Distribution Analysis** — Scatter plots and histograms to characterize corruption patterns

### Model Architecture

The model is a **DnCNN-style U-Net** — a symmetric encoder-decoder with skip connections and residual blocks:

```
┌─────────────────────────────────────────────────┐
│                  DnCNN-style U-Net               │
├─────────────────────────────────────────────────┤
│                                                  │
│  INPUT (3, 32, 32)                               │
│    │                                             │
│    ▼                                             │
│  ┌──────────────┐                                │
│  │  Encoder E1   │ Conv → BN → ReLU + ResBlock   │
│  │  (3 → 128)   │ Output: (128, 32, 32)         │
│  └──────┬───────┘                                │
│         │ ──────────────────────── skip₁          │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │  Encoder E2   │ MaxPool + Conv + ResBlock      │
│  │ (128 → 256)  │ Output: (256, 16, 16)         │
│  └──────┬───────┘                                │
│         │ ──────────────────────── skip₂          │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │  Encoder E3   │ MaxPool + Conv + ResBlock      │
│  │ (256 → 512)  │ Output: (512, 8, 8)           │
│  └──────┬───────┘                                │
│         │ ──────────────────────── skip₃          │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │  Bottleneck   │ MaxPool + Conv + ResBlock      │
│  │ (512 → 1024) │ Output: (1024, 4, 4)          │
│  └──────┬───────┘                                │
│         │                                        │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │  Decoder D3   │ Upsample + Cat(skip₃) + Conv  │
│  │(1024+512→512)│ + ResBlock                     │
│  └──────┬───────┘                                │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │  Decoder D2   │ Upsample + Cat(skip₂) + Conv  │
│  │(512+256→256) │ + ResBlock                     │
│  └──────┬───────┘                                │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │  Decoder D1   │ Upsample + Cat(skip₁) + Conv  │
│  │(256+128→128) │ + ResBlock                     │
│  └──────┬───────┘                                │
│         │                                        │
│         ▼                                        │
│  ┌──────────────┐                                │
│  │   Head        │ Conv(128 → 3) + Tanh          │
│  └──────────────┘                                │
│                                                  │
│  OUTPUT (3, 32, 32) in [-1, 1]                   │
│                                                  │
│  Total Parameters: 77,693,059                    │
└─────────────────────────────────────────────────┘
```

**Key design choices:**
- **Residual Blocks** — Each block uses two 3×3 convolutions with BatchNorm and ReLU, with a residual/skip addition for stable gradient flow
- **Skip Connections** — Encoder features are concatenated with decoder features at each level, preserving fine-grained spatial information
- **Tanh Output** — Maps predictions to [-1, 1] range matching the normalized input
- **77.7M Parameters** — A large-capacity model to handle diverse corruption patterns

### Loss Function

A **combined loss** that balances two objectives:

```python
Loss = α × L1(pred, target) + (1 - α) × MSE(pred, target)
```

| Component | Purpose | Weight (α=0.5) |
|:----------|:--------|:---------------|
| **L1 Loss** | Preserves sharp edges, reduces blurring | 50% |
| **MSE Loss** | Directly optimizes the competition metric | 50% |

### Training Strategy

| Hyperparameter     | Value                                    |
| :----------------- | :--------------------------------------- |
| **Optimizer**      | AdamW (weight_decay=1e-4)                |
| **Learning Rate**  | Cyclic LR: 3e-4 → 3e-3 (triangular)     |
| **Batch Size**     | 128                                      |
| **Epochs**         | 45                                       |
| **Augmentation**   | Random horizontal flip                   |
| **Split**          | 44,000 train / 4,000 validation          |
| **Checkpointing**  | Best model saved by validation MSE       |

The **Cyclic Learning Rate** scheduler oscillates between a lower bound (3e-4) and upper bound (3e-3) with a triangular pattern, helping the model escape local minima and explore the loss landscape more effectively.

### Test-Time Augmentation

At inference, predictions are averaged over the original input and its **horizontally flipped** version:

```python
prediction = (model(input) + flip(model(flip(input)))) / 2
```

This simple TTA provides an additional **~5% MSE reduction** (1142.79 → 1085.95).

---

## 📊 Results

### Performance Summary

| Metric                    | Value         |
| :------------------------ | :------------ |
| Baseline MSE (corrupted)  | 4822.19       |
| Model MSE (no TTA)        | 1142.79       |
| **Model MSE (with TTA)**  | **1085.95**   |
| **Improvement**           | **76.3%**     |

### Training Curves

The model converges smoothly over 45 epochs with consistent improvement on both training and validation sets:

- **Loss Curve**: Both training and validation losses decrease steadily, showing good generalization
- **MSE Curve**: The competition metric (MSE on 0-255 scale) drops rapidly in early epochs and continues to improve

### Visual Results

The notebook includes visual comparisons showing:
- **Row 1**: Corrupted input images
- **Row 2**: Model predictions (reconstructed)
- **Row 3**: Ground truth clean images
- **Row 4**: Absolute difference maps (prediction vs. ground truth)

The model effectively handles diverse corruption types including color distortion, occlusion, and noise.

---

## 📁 Project Structure

```
image_reconstruction/
├── ImageConstruction.ipynb    # Main notebook with full pipeline
├── README.md                  # This file
├── best_model.pth             # Saved model checkpoint (generated during training)
└── submission.csv             # Final submission file (12,000 rows × 3,073 cols)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib opencv-python pillow tqdm
```

### Running the Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pranavgore676/Image_Reconstruction.git
   cd Image_Reconstruction
   ```

2. **Download the dataset** from the Kaggle competition and place it in the `Datasets/` directory.

3. **Open and run the notebook:**
   ```bash
   jupyter notebook ImageConstruction.ipynb
   ```

4. The notebook will:
   - Load and explore the dataset
   - Profile corruption types via EDA
   - Build and train the DnCNN-style U-Net
   - Evaluate on validation set with visual comparisons
   - Generate the submission CSV with TTA

### Hardware Requirements

| Component | Recommendation                        |
| :-------- | :------------------------------------ |
| **GPU**   | NVIDIA GPU with ≥8GB VRAM (CUDA)      |
| **RAM**   | ≥16GB system memory                   |
| **Disk**  | ≥2GB free space for dataset + model   |

> 💡 **Tip:** The notebook was developed on Kaggle with GPU acceleration. For local execution, ensure CUDA is properly configured.

---

## 🧠 Technical Highlights

- **DnCNN-style Residual Blocks** for stable training of deep networks
- **U-Net skip connections** preserve spatial detail across scales
- **Combined L1 + MSE loss** balances edge preservation with metric optimization
- **Cyclic Learning Rate** for better convergence and generalization
- **Data augmentation** (random horizontal flips) for training regularization
- **Test-Time Augmentation** for improved prediction quality at inference

---

## 📝 Acknowledgements

- Dataset from [Kaggle Image Reconstruction Under Corruption Competition](https://www.kaggle.com/competitions/image-reconstruction-under-corruption)
- Architecture inspired by [DnCNN](https://arxiv.org/abs/1608.03981) and [U-Net](https://arxiv.org/abs/1505.04597)
- Built with [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/), and [NumPy](https://numpy.org/)

---

<p align="center">
  <b>⭐ If you found this useful, please consider giving the repo a star!</b>
</p>
