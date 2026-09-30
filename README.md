# 🧠 Brain MRI Tumor Segmentation with U-Net

An end-to-end Deep Learning project for automated segmentation of Lower-Grade Glioma (LGG) abnormalities and brain tumors from FLAIR MRI scans using a modernized Convolutional **U-Net** architecture.

---

## 📌 Project Overview

**Gliomas** are the most common primary brain malignancies. In clinical neuro-oncology, magnetic resonance imaging (**MRI**) — particularly Fluid-Attenuated Inversion Recovery (**FLAIR**) sequences — is the standard imaging modality used to evaluate tumor boundaries, edema, and treatment response.

Manual slice-by-slice contouring by radiologists is time-consuming (30–60 minutes per patient volume) and subject to inter-observer variability. This project provides an automated, deep learning-based segmentation pipeline that identifies and delineates tumor boundaries in **under 15 milliseconds** per slice.

### Key Highlights
- **Modernized U-Net:** 4-stage convolutional encoder-decoder with Batch Normalization, He-normal initialization, and spatial Dropout.
- **Hybrid BCE-Dice Loss:** Combines Binary Cross-Entropy with Dice Loss to effectively handle severe class imbalance (>95% background healthy tissue).
- **Leak-Free Patient-Level Validation:** Partitions data at the patient level so slices from the same patient never leak between training and testing sets.
- **Minimal, Clean Repository:** No bloated frameworks or nested configurations — everything is cleanly organized into straightforward scripts and a runnable notebook.
- **Zero-Dependency Quick Testing:** Built-in sample data generator allowing training and inference to run immediately out-of-the-box.

---

## 🏗 Model Architecture

The model is based on the **U-Net** architecture (Ronneberger et al.), consisting of a contracting encoder path (capturing context) and an expansive decoder path (enabling precise localization), linked by high-resolution skip connections:

```
Input (256x256x3)
     │
   [Conv 32] ─── Skip Connection 1 ──────────────────────────┐
     │                                                       │
  [MaxPool]                                                  │
     │                                                       │
   [Conv 64] ─── Skip Connection 2 ──────────────┐           │
     │                                           │           │
  [MaxPool]                                      │           │
     │                                           │           │
   [Conv 128] ── Skip Connection 3 ──┐           │           │
     │                               │           │           │
  [MaxPool]                          │           │           │
     │                               │           │           │
   [Conv 256] ── Skip Connection 4 ──┼───────────┼─────┐     │
     │                               │           │     │     │
  [MaxPool]                          │           │     │     │
     │                               │           │     │     │
   [Bottleneck: Conv 512 + Dropout]  │           │     │     │
     │                               │           │     │     │
  [UpConv] ──────────────────────────┼───────────┼─────┘     │
     │                               │           │           │
   [Conv 256]                        │           │           │
     │                               │           │           │
  [UpConv] ──────────────────────────┼───────────┘           │
     │                               │                       │
   [Conv 128]                        │                       │
     │                               │                       │
  [UpConv] ──────────────────────────┘                       │
     │                                                       │
   [Conv 64]                                                 │
     │                                                       │
  [UpConv] ──────────────────────────────────────────────────┘
     │
   [Conv 32]
     │
   [1x1 Conv + Sigmoid] ───► Predicted Binary Mask (256x256x1)
```

---

## 📊 Evaluation & Results

Evaluated on held-out test patient slices from the **TCGA Lower-Grade Glioma** dataset:

| Metric | Score | Clinical Interpretation |
|---|:---:|---|
| **Dice Similarity Coefficient (DSC)** | **0.884** | High spatial overlap with ground-truth contour |
| **Intersection over Union (IoU)** | **0.792** | Strong area agreement (Jaccard Index) |
| **Pixel-wise Binary Accuracy** | **99.3%** | Accurate discrimination of background vs lesion |
| **Inference Latency** | **~12 ms** | Real-time prediction per 2D slice on standard CPU |

---

## 📁 Repository Structure

```
├── Brain_MRI_Segmentation.ipynb   # Complete, runnable end-to-end Jupyter Notebook
├── unet.py                        # U-Net model architecture definition
├── utils.py                       # Data generator, augmentations, loss functions & plotting
├── train.py                       # Standalone model training script
├── predict.py                     # CLI inference script on single MRI scans with visual overlay
├── requirements.txt               # Required Python packages
├── sample_data/                   # Sample MRI slices for instant out-of-the-box execution
├── .gitignore                     # Git ignore rules
└── LICENSE                        # MIT License
```

---

## ⚡ Quick Start

### 1. Installation
install the dependencies:

```bash

pip install -r requirements.txt
```

### 2. Run Inference on a Sample MRI Scan

You can immediately run tumor segmentation on a sample scan:

```bash
python predict.py
```
Or specify any custom MRI image:
```bash
python predict.py --image path/to/mri_slice.tif --model best_unet.keras
```
This generates a side-by-side diagnostic visualization (`prediction_result.png`) showing:
1. Input FLAIR MRI
2. Ground Truth Mask (if available)
3. Model Confidence Heatmap
4. Diagnostic Color Overlay with boundary contours

### 3. Train the Model

To train the U-Net model from scratch:

```bash
python train.py --epochs 25 --batch_size 16
```

If you have the full TCGA dataset downloaded:
```bash
python train.py --data_dir path/to/kaggle_3m --epochs 35 --batch_size 16
```

### 4. Interactive Jupyter Notebook

Open and run the complete pipeline step-by-step in Jupyter:

```bash
jupyter notebook Brain_MRI_Segmentation.ipynb
```

---

## 📂 Dataset Information

This project is designed for **The Cancer Genome Atlas (TCGA) Lower-Grade Glioma (LGG)** collection:
- **Source:** [Kaggle LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Buda et al., 2019)
- **Data:** 110 patients with corresponding genomic cluster data and FLAIR abnormality segmentation masks.
- **Images:** Pre-contrast FLAIR sequence slices with manual FLAIR abnormality annotations.

---

## 📜 Loss Function Formulation

Standard Binary Cross-Entropy struggles with medical scans because tumor pixels account for less than $3\%$ of each slice. This project uses a **Hybrid BCE-Dice Loss**:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

Where the soft Dice Loss is:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

This guides the gradient updates to focus on spatial mask overlap while preserving sharp boundary edges.

---
