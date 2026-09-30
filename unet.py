"""
U-Net Architecture for Brain MRI Segmentation
Author: Himanshu

Implements a deep Convolutional U-Net neural network with Batch Normalization
and Spatial Dropout for segmenting brain tumors / FLAIR abnormalities from MRI scans.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(inputs, filters, dropout_rate=0.0):
    """
    Two consecutive 3x3 convolutional layers with Batch Normalization and ReLU.
    """
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def build_unet(input_shape=(256, 256, 3), start_neurons=32, dropout_rate=0.2):
    """
    Builds a 4-level U-Net model with skip connections.

    Parameters:
    -----------
    input_shape : tuple
        Dimensions of input MRI slices (height, width, channels). Default: (256, 256, 3).
    start_neurons : int
        Base filter count at the first level (32 -> 64 -> 128 -> 256 -> 512).
    dropout_rate : float
        Dropout probability applied in the deeper layers and bottleneck.

    Returns:
    --------
    model : tf.keras.models.Model
        Constructed Keras U-Net model.
    """
    inputs = layers.Input(shape=input_shape, name="mri_input")

    # --- Contracting Path (Encoder) ---
    c1 = conv_block(inputs, start_neurons * 1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, start_neurons * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, start_neurons * 4, dropout_rate=dropout_rate / 2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, start_neurons * 8, dropout_rate=dropout_rate)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # --- Bottleneck ---
    b = conv_block(p4, start_neurons * 16, dropout_rate=dropout_rate)

    # --- Expansive Path (Decoder) ---
    u6 = layers.Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), padding="same")(b)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, start_neurons * 8, dropout_rate=dropout_rate)

    u7 = layers.Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, start_neurons * 4, dropout_rate=dropout_rate / 2)

    u8 = layers.Conv2DTranspose(start_neurons * 2, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, start_neurons * 2)

    u9 = layers.Conv2DTranspose(start_neurons * 1, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, start_neurons * 1)

    # --- Output Layer ---
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="tumor_mask")(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="Brain_MRI_UNet")
    return model


if __name__ == "__main__":
    # Test model compilation and forward pass shape
    model = build_unet()
    model.summary()
    print("\nModel constructed successfully! Parameters:", model.count_params())
