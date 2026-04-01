"""
Training Pipeline — MediVision AI
Run: python src/train.py --model vgg16 --epochs 50
"""

import argparse
import os
import sys
import yaml
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_cnn import build_custom_cnn, run_hyperparameter_search
from models.vgg16_transfer import (build_vgg16_model, compile_model,
                                    get_callbacks, compare_pretrained_vs_scratch)
from models.dcgan import MedicalDCGAN
from utils.data_loader import create_dataset_from_directory, get_class_weights
from utils.visualization import plot_confusion_matrix


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def train_cnn(config, epochs, tune_hyperparams=False):
    """Train custom CNN with optional Keras Tuner search."""
    print("\n" + "="*60)
    print("  TRAINING: Custom CNN")
    print("="*60)

    data_dir = config['data']['dataset_path']
    train_ds = create_dataset_from_directory(data_dir, split='train')
    val_ds = create_dataset_from_directory(data_dir, split='val')
    class_weights = get_class_weights(data_dir)

    with mlflow.start_run(run_name=f"CNN_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_param("model", "custom_cnn")
        mlflow.log_param("epochs", epochs)
        mlflow.log_params(config['cnn'])

        if tune_hyperparams:
            print("\n🔍 Running Keras Tuner hyperparameter search...")
            model, best_hps = run_hyperparameter_search(train_ds, val_ds)
            for k, v in best_hps.values.items():
                mlflow.log_param(f"best_{k}", v)
        else:
            model = build_custom_cnn(num_classes=config['data']['num_classes'])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall()]
            )

        model.summary()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "checkpoints/cnn_best.h5", monitor='val_accuracy',
                save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir="logs/cnn/")
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # Log metrics
        best_acc = max(history.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_acc)
        mlflow.keras.log_model(model, "cnn_model")

        print(f"\n✅ CNN Training Complete | Best Val Accuracy: {best_acc:.4f}")
        plot_training_history(history, "CNN")
        return model, history


def train_vgg16(config, epochs, compare_weights=False):
    """Train VGG16 with optional pretrained vs scratch comparison."""
    print("\n" + "="*60)
    print("  TRAINING: VGG16 Transfer Learning")
    print("="*60)

    data_dir = config['data']['dataset_path']
    train_ds = create_dataset_from_directory(data_dir, split='train')
    val_ds = create_dataset_from_directory(data_dir, split='val')

    with mlflow.start_run(run_name=f"VGG16_{datetime.now().strftime('%H%M%S')}"):
        mlflow.log_param("model", "vgg16_transfer")
        mlflow.log_param("epochs", epochs)

        if compare_weights:
            # Experiment 6: Compare with/without pretrained weights
            results = compare_pretrained_vs_scratch(train_ds, val_ds, epochs=epochs)
            for tag, res in results.items():
                mlflow.log_metric(f"vgg16_{tag}_val_accuracy", res['final_val_acc'])
            return results

        # Default: use pretrained weights
        model, base = build_vgg16_model(use_pretrained_weights=True)
        model = compile_model(model)
        model.summary()

        # Phase 1: Train classifier head only
        print("\n📌 Phase 1: Training classifier head...")
        history1 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs // 2,
            callbacks=get_callbacks("vgg16_phase1")
        )

        # Phase 2: Fine-tune top layers
        from models.vgg16_transfer import fine_tune_model
        model, history2 = fine_tune_model(model, base, train_ds, val_ds,
                                           fine_tune_epochs=epochs // 2)

        best_acc = max(history2.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_acc)
        mlflow.keras.log_model(model, "vgg16_model")

        print(f"\n✅ VGG16 Training Complete | Best Val Accuracy: {best_acc:.4f}")
        return model


def train_gan(config, epochs):
    """Train DCGAN for synthetic MRI generation."""
    print("\n" + "="*60)
    print("  TRAINING: DCGAN for Medical Image Synthesis")
    print("="*60)

    data_dir = config['data']['dataset_path']

    # Load grayscale MRI images for GAN training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        label_mode=None,
        color_mode='grayscale',
        image_size=(64, 64),
        batch_size=32
    )
    train_ds = train_ds.map(
        lambda x: (tf.cast(x, tf.float32) / 127.5) - 1.0
    ).prefetch(tf.data.AUTOTUNE)

    gan = MedicalDCGAN(latent_dim=config['dcgan']['latent_dim'],
                        img_shape=(64, 64, 1))
    gan.generator.summary()
    gan.discriminator.summary()

    gen_losses, disc_losses = gan.train(train_ds, epochs=epochs)
    gan.plot_training_curves()

    print(f"\n✅ GAN Training Complete | Generated images in gan_outputs/")
    return gan


def plot_training_history(history, model_name):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train', color='#3498db')
    ax1.plot(history.history['val_accuracy'], label='Validation', color='#e74c3c')
    ax1.set_title(f'{model_name} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', color='#3498db')
    ax2.plot(history.history['val_loss'], label='Validation', color='#e74c3c')
    ax2.set_title(f'{model_name} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name.lower()}_training_curves.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='MediVision AI Training Pipeline')
    parser.add_argument('--model', choices=['cnn', 'vgg16', 'dcgan', 'all'],
                        default='vgg16', help='Model to train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter search (CNN only)')
    parser.add_argument('--compare-weights', action='store_true',
                        help='Compare pretrained vs random VGG16 weights')
    args = parser.parse_args()

    config = load_config()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    print(f"\n🚀 MediVision AI Training Pipeline")
    print(f"   Model: {args.model} | Epochs: {args.epochs}")
    print(f"   MLflow tracking: {config['mlflow']['tracking_uri']}\n")

    if args.model == 'cnn':
        train_cnn(config, args.epochs, tune_hyperparams=args.tune)
    elif args.model == 'vgg16':
        train_vgg16(config, args.epochs, compare_weights=args.compare_weights)
    elif args.model == 'dcgan':
        train_gan(config, args.epochs)
    elif args.model == 'all':
        train_cnn(config, args.epochs)
        train_vgg16(config, args.epochs)
        train_gan(config, epochs=200)


if __name__ == "__main__":
    main()
