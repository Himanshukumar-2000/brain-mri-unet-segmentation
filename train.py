"""
Training Pipeline for Brain MRI Tumor Segmentation
Author: Himanshu

Trains a deep U-Net model on FLAIR Brain MRI slices, evaluates metrics on a held-out
test set, and saves both model checkpoints and visual prediction galleries.

Usage:
------
python train.py
python train.py --epochs 30 --batch_size 16 --data_dir sample_data
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from unet import build_unet
from utils import (
    DataGenerator,
    bce_dice_loss,
    dice_coef,
    iou_metric,
    load_dataset_paths,
    plot_training_history,
    split_data,
    visualize_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on Brain MRI Scans")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing MRI patient scans")
    parser.add_argument("--checkpoint", type=str, default="best_unet.keras", help="Path to save best weights")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("      BRAIN MRI GLIOMA SEGMENTATION - TRAINING")
    print("=" * 65)

    # 1. Locate Dataset
    if args.data_dir and os.path.exists(args.data_dir):
        data_dir = args.data_dir
    elif os.path.exists("kaggle_3m"):
        data_dir = "kaggle_3m"
    elif os.path.exists("sample_data"):
        data_dir = "sample_data"
    else:
        raise FileNotFoundError("No dataset directory found. Please specify --data_dir.")

    print(f"Loading MRI scans from: {data_dir}")
    df = load_dataset_paths(data_dir)
    print(f"Total slices: {len(df)} | Patients: {df['patient_id'].nunique()} | Tumor slices: {df['has_tumor'].sum()}")

    # 2. Split Data (Leak-free by patient)
    df_train, df_val, df_test = split_data(df, test_size=0.15, val_size=0.15, random_state=42)
    print(f"Dataset split -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # 3. Create Generators
    train_gen = DataGenerator(df_train, batch_size=args.batch_size, augment=True, shuffle=True)
    val_gen = DataGenerator(df_val, batch_size=args.batch_size, augment=False, shuffle=False)
    test_gen = DataGenerator(df_test, batch_size=args.batch_size, augment=False, shuffle=False)

    # 4. Build Model
    model = build_unet(input_shape=(256, 256, 3), start_neurons=32, dropout_rate=0.2)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_metric, "binary_accuracy"],
    )

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(args.checkpoint, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_dice_coef", mode="max", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    # 6. Train Model
    print(f"\nStarting training for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate on Test Set
    print("\nEvaluating model on test set...")
    results = model.evaluate(test_gen, verbose=1)
    print("\n" + "=" * 45)
    print(f"Test Loss        : {results[0]:.4f}")
    print(f"Test Dice Score  : {results[1]:.4f}")
    print(f"Test IoU Score   : {results[2]:.4f}")
    print(f"Test Accuracy    : {results[3]:.4f}")
    print("=" * 45)

    # 8. Save Visual Diagnostics
    plot_training_history(history, save_path="training_curves.png")
    visualize_predictions(model, df_test, num_samples=min(4, len(df_test)), save_path="predictions_grid.png")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
