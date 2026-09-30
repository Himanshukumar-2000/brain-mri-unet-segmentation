"""
Utility Functions for Brain MRI Segmentation
Author: Himanshu

Contains data loading, preprocessing, augmentation generator,
loss functions (Dice, IoU, BCE-Dice), and visualization helpers.
"""

import os
from glob import glob
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. EVALUATION METRICS & LOSS FUNCTIONS
# ==============================================================================

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Computes the Sørensen-Dice Similarity Coefficient (DSC).
    Measures spatial overlap between ground truth and predicted binary mask.
    """
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss function (1 - Dice Coefficient).
    """
    return 1.0 - dice_coef(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU / Jaccard Index).
    """
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(y_true, y_pred):
    """
    Hybrid loss: Combines Binary Cross-Entropy with Dice Loss.
    Addresses extreme class imbalance while preserving sharp edge boundaries.
    """
    y_true_f = K.cast(y_true, "float32")
    y_pred_f = K.clip(K.cast(y_pred, "float32"), 1e-7, 1.0 - 1e-7)

    bce = K.mean(-y_true_f * K.log(y_pred_f) - (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    dice = dice_loss(y_true_f, y_pred_f)
    return 0.5 * bce + 0.5 * dice


# ==============================================================================
# 2. DATASET SCANNING & SPLITTING
# ==============================================================================

def load_dataset_paths(data_dir):
    """
    Scans directory for MRI slices and corresponding '_mask' files.
    Returns a sorted DataFrame with image paths, mask paths, and tumor flags.
    """
    mask_files = glob(os.path.join(data_dir, "**", "*_mask*.tif"), recursive=True)
    if not mask_files:
        mask_files = glob(os.path.join(data_dir, "**", "*_mask*.png"), recursive=True)

    records = []
    for mask_path in mask_files:
        norm_mask = os.path.normpath(mask_path)
        img_path = norm_mask.replace("_mask", "")
        if not os.path.exists(img_path):
            continue

        patient_id = os.path.basename(os.path.dirname(norm_mask))
        records.append({
            "image_path": img_path,
            "mask_path": norm_mask,
            "patient_id": patient_id,
            "filename": os.path.basename(img_path)
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise ValueError(f"No valid image/mask pairs found in '{data_dir}'.")

    # Flag whether slice contains a tumor
    has_tumor = []
    for m in df["mask_path"]:
        mask_img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        has_tumor.append(bool(mask_img is not None and np.max(mask_img) > 0))
    df["has_tumor"] = has_tumor

    return df.sort_values(by=["patient_id", "filename"]).reset_index(drop=True)


def split_data(df, test_size=0.10, val_size=0.20, random_state=42):
    """
    Splits the dataset into train, validation, and test subsets.
    If multiple patients exist, splits by patient ID to prevent slice leakage.
    """
    patients = df["patient_id"].unique()
    if len(patients) >= 3:
        p_train_val, p_test = train_test_split(patients, test_size=test_size, random_state=random_state)
        p_train, p_val = train_test_split(p_train_val, test_size=val_size, random_state=random_state)

        df_train = df[df["patient_id"].isin(p_train)].reset_index(drop=True)
        df_val = df[df["patient_id"].isin(p_val)].reset_index(drop=True)
        df_test = df[df["patient_id"].isin(p_test)].reset_index(drop=True)
    else:
        df_train_val, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
        df_train, df_val = train_test_split(df_train_val, test_size=val_size, random_state=random_state)
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

    return df_train, df_val, df_test


# ==============================================================================
# 3. DATA GENERATOR WITH AUGMENTATION
# ==============================================================================

class DataGenerator(Sequence):
    """
    Keras Sequence Data Generator for loading MRI images and masks in batches.
    Includes data normalization and on-the-fly geometric augmentations.
    """

    def __init__(self, df, batch_size=16, target_size=(256, 256), augment=False, shuffle=True):
        self.df = df.copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        images, masks = [], []
        for _, row in batch_df.iterrows():
            img, mask = self._load_pair(row["image_path"], row["mask_path"])
            if self.augment:
                img, mask = self._augment(img, mask)
            images.append(img)
            masks.append(mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

    def _load_pair(self, img_path, mask_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        return img, mask

    def _augment(self, img, mask):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        # Random rotation (-15 to 15 degrees)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = self.target_size
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask_2d = cv2.warpAffine(mask.squeeze(), matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = np.expand_dims((mask_2d > 0.5).astype(np.float32), axis=-1)

        # Random brightness scaling
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.85, 1.15)
            img = np.clip(img**gamma, 0.0, 1.0)

        return img, mask


# ==============================================================================
# 4. VISUALIZATION & PREDICTION HELPERS
# ==============================================================================

def create_overlay(image, mask, alpha=0.45, color=(255, 40, 40)):
    """
    Overlays a colored binary mask with boundary contours onto an MRI image.
    """
    if image.max() <= 1.0:
        base = (image * 255.0).astype(np.uint8)
    else:
        base = image.astype(np.uint8)

    m = np.squeeze(mask) > 0.5
    colored = np.zeros_like(base)
    for c in range(3):
        colored[:, :, c] = m * color[c]

    blended = np.where(m[:, :, None], cv2.addWeighted(base, 1.0 - alpha, colored, alpha, 0), base)

    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 2)
    return blended


def plot_training_history(history, save_path="training_curves.png"):
    """
    Plots training and validation loss and Dice coefficient curves.
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, hist["loss"], "o-", label="Train Loss", color="#1f77b4")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], "s--", label="Val Loss", color="#d62728")
    axes[0].set_title("Loss Across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Dice coefficient curve
    dice_key = "dice_coef" if "dice_coef" in hist else "dice_coefficient"
    val_dice_key = f"val_{dice_key}"
    if dice_key in hist:
        axes[1].plot(epochs, hist[dice_key], "o-", label="Train Dice", color="#2ca02c")
        if val_dice_key in hist:
            axes[1].plot(epochs, hist[val_dice_key], "s--", label="Val Dice", color="#ff7f0e")
    axes[1].set_title("Dice Similarity Coefficient", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Training curves saved to {save_path}")
    plt.close()


def visualize_predictions(model, df_test, num_samples=4, threshold=0.5, save_path="predictions_grid.png"):
    """
    Displays test slices with Ground Truth and Model Predictions side-by-side.
    """
    num_samples = min(num_samples, len(df_test))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        row = df_test.iloc[i]
        img = cv2.imread(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = cv2.resize(img, (256, 256)) / 255.0

        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask_norm = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 127

        # Predict
        pred = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0, :, :, 0]
        binary_pred = pred > threshold
        overlay = create_overlay(img_norm, binary_pred)

        # Plot
        axes[i, 0].imshow(img_norm)
        axes[i, 0].set_title("Input MRI", fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_norm, cmap="inferno")
        axes[i, 1].set_title("Ground Truth", fontweight="bold")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="inferno", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (Max: {pred.max():.2f})", fontweight="bold")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Diagnostic Overlay", fontweight="bold")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Prediction visualization saved to {save_path}")
    plt.close()
