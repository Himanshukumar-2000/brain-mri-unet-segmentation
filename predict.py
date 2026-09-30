"""
Inference Script for Brain MRI Tumor Segmentation
Author: Himanshu

Runs model inference on an MRI scan slice, calculates tumor area,
and produces a clinical side-by-side diagnostic visualization with color overlay.

Usage:
------
python predict.py --image sample_data/TCGA_SYNTH_01/TCGA_SYNTH_01_4.tif
"""

import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from unet import build_unet
from utils import bce_dice_loss, create_overlay, dice_coef, iou_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Tumor Mask on Brain MRI Slice")
    parser.add_argument("--image", type=str, default=None, help="Path to input MRI image (.tif, .png, .jpg)")
    parser.add_argument("--model", type=str, default="best_unet.keras", help="Path to saved model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for lesion detection")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Path to save diagnostic image")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Select Image
    if args.image and os.path.exists(args.image):
        image_path = args.image
    else:
        # Default fallback to first sample slice found
        from glob import glob
        samples = [f for f in glob("sample_data/**/*.tif", recursive=True) if "_mask" not in f]
        if samples:
            image_path = samples[0]
            print(f"No image specified. Using sample image: {image_path}")
        else:
            raise FileNotFoundError("No image provided and no sample data found.")

    # 2. Load Model
    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou_metric": iou_metric,
    }

    if os.path.exists(args.model):
        print(f"Loading trained weights from: {args.model}")
        model = load_model(args.model, custom_objects=custom_objects, compile=False)
    else:
        print(f"[Notice] Checkpoint '{args.model}' not found. Initializing architecture for demonstration.")
        model = build_unet(input_shape=(256, 256, 3))

    # 3. Read and Preprocess Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = cv2.resize(img_rgb, (256, 256)) / 255.0

    # 4. Predict
    inp = np.expand_dims(img_norm, axis=0)
    pred_prob = model.predict(inp, verbose=0)[0, :, :, 0]
    binary_pred = pred_prob >= args.threshold

    # 5. Diagnostic Metrics
    tumor_pixels = int(np.sum(binary_pred))
    slice_pixels = binary_pred.size
    coverage_pct = (tumor_pixels / slice_pixels) * 100.0
    has_lesion = bool(tumor_pixels > 10)

    print("=" * 55)
    print("           MRI SEGMENTATION REPORT")
    print("=" * 55)
    print(f"File Name        : {os.path.basename(image_path)}")
    print(f"Tumor Status     : {'POSITIVE (Lesion Detected)' if has_lesion else 'NEGATIVE (Normal Tissue)'}")
    print(f"Lesion Area      : {tumor_pixels} pixels ({coverage_pct:.2f}% of slice)")
    print(f"Peak Confidence  : {float(np.max(pred_prob)):.2%}")
    print("=" * 55)

    # 6. Check for Ground Truth
    gt_path = image_path.replace(".tif", "_mask.tif").replace(".png", "_mask.png")
    has_gt = os.path.exists(gt_path)

    cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    # Input image
    axes[0].imshow(img_norm)
    axes[0].set_title("Input FLAIR MRI", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    col_idx = 1
    if has_gt:
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_norm = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
        axes[col_idx].imshow(gt_norm, cmap="inferno")
        axes[col_idx].set_title("Ground Truth", fontsize=11, fontweight="bold")
        axes[col_idx].axis("off")
        col_idx += 1

    # Probability heatmap
    im = axes[col_idx].imshow(pred_prob, cmap="inferno", vmin=0, vmax=1)
    axes[col_idx].set_title(f"Confidence Map (Max: {pred_prob.max():.2f})", fontsize=11, fontweight="bold")
    axes[col_idx].axis("off")
    col_idx += 1

    # Clinical overlay
    overlay = create_overlay(img_norm, binary_pred, alpha=0.45, color=(255, 40, 40))
    axes[col_idx].imshow(overlay)
    status_label = "POSITIVE" if has_lesion else "NEGATIVE"
    axes[col_idx].set_title(f"Clinical Overlay ({status_label})", fontsize=11, fontweight="bold")
    axes[col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Diagnostic visualization saved to: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
