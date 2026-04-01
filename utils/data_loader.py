"""
Data Loading, Preprocessing, and Augmentation Pipeline
"""

import tensorflow as tf
import numpy as np
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_COLORS = {'glioma': '#e74c3c', 'meningioma': '#f39c12',
                'notumor': '#2ecc71', 'pituitary': '#3498db'}


def create_dataset_from_directory(data_dir, img_size=(224, 224), batch_size=32,
                                   split='train'):
    augment = (split == 'train')

    dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, split),
        labels='inferred',
        label_mode='categorical',
        class_names=CLASSES,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        seed=42
    )

    dataset = dataset.map(
        lambda x, y: (preprocess_batch(x, augment=augment), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def preprocess_batch(images, augment=False):
    """Normalize and optionally augment a batch of images."""
    images = tf.cast(images, tf.float32)
    images = images / 127.5 - 1.0  # Scale to [-1, 1]

    if augment:
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        images = tf.image.random_brightness(images, max_delta=0.1)
        images = tf.image.random_contrast(images, lower=0.9, upper=1.1)

    return images


def preprocess(image, augment=False):
    """Single image preprocessing for inference."""
    image = tf.cast(image, tf.float32)
    image = image / 127.5 - 1.0
    return image


def get_class_weights(data_dir):
    """Compute class weights to handle class imbalance."""
    counts = {}
    for cls in CLASSES:
        path = Path(data_dir) / 'train' / cls
        counts[cls] = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png'))) + len(list(path.glob('*.jpeg')))

    total = sum(counts.values())
    weights = {i: total / (len(CLASSES) * max(count, 1))
               for i, (cls, count) in enumerate(counts.items())}

    print("\nClass Distribution:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>15}: {counts[cls]:>5} images | weight: {weights[i]:.3f}")

    return weights


def prepare_image_for_inference(image_path, img_size=(224, 224)):
    """Load and preprocess a single image for model inference."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    img = tf.expand_dims(img, axis=0)
    return img


if __name__ == "__main__":
    print("Data loader module ready")
    print(f"Classes: {CLASSES}")
