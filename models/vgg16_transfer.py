"""
VGG16 Transfer Learning for Brain Tumor Classification
Covers: Experiment 6 - Pre-trained VGG16 with and without pre-defined weights
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import yaml


def build_vgg16_model(use_pretrained_weights=True, freeze_layers=15,
                      num_classes=4, input_shape=(224, 224, 3)):
    """
    Build VGG16 model with or without pretrained ImageNet weights.
    
    Args:
        use_pretrained_weights (bool): If True, loads ImageNet weights (Experiment 6a)
                                      If False, trains from scratch (Experiment 6b)
        freeze_layers (int): Number of base layers to freeze during initial training
        num_classes (int): Number of output classes
    """
    weights = 'imagenet' if use_pretrained_weights else None
    tag = "with_pretrained_weights" if use_pretrained_weights else "from_scratch"
    print(f"\n🔧 Building VGG16 {tag}...")

    # Load VGG16 base
    base_model = VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )

    # Freeze layers strategy
    if use_pretrained_weights and freeze_layers > 0:
        for layer in base_model.layers[:freeze_layers]:
            layer.trainable = False
        print(f"   Frozen first {freeze_layers} layers of VGG16")

    # Custom classifier head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=base_model.input, outputs=output,
                         name=f"VGG16_{tag}")

    return model, base_model


def compile_model(model, learning_rate=0.001, fine_tune=False):
    """Compile with appropriate LR — lower for fine-tuning."""
    lr = 0.0001 if fine_tune else learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


def fine_tune_model(model, base_model, train_data, val_data,
                    unfreeze_from=10, fine_tune_epochs=20):
    """
    Two-phase training:
    Phase 1: Train only classifier head
    Phase 2: Unfreeze top layers and fine-tune with low LR
    """
    print("\n📌 Phase 2: Fine-tuning top layers...")

    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    # Recompile with lower learning rate
    model = compile_model(model, fine_tune=True)

    callbacks = get_callbacks("vgg16_finetuned")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=fine_tune_epochs,
        callbacks=callbacks
    )

    return model, history


def get_callbacks(model_name):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            f"checkpoints/{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=f"logs/{model_name}/")
    ]


def compare_pretrained_vs_scratch(train_data, val_data, epochs=30):
    """
    Core experiment: Compare VGG16 with pretrained vs random weights.
    Directly implements Experiment 6 requirement.
    """
    results = {}

    for use_weights in [True, False]:
        tag = "pretrained" if use_weights else "scratch"
        print(f"\n{'='*50}")
        print(f"Training VGG16 {tag.upper()}")
        print('='*50)

        model, base = build_vgg16_model(use_pretrained_weights=use_weights)
        model = compile_model(model)

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=get_callbacks(f"vgg16_{tag}")
        )

        results[tag] = {
            'model': model,
            'history': history,
            'final_val_acc': max(history.history['val_accuracy'])
        }

        print(f"\n✅ VGG16 ({tag}) best val accuracy: "
              f"{results[tag]['final_val_acc']:.4f}")

    print("\n📊 Comparison Summary:")
    for tag, res in results.items():
        print(f"  VGG16 ({tag:>10}): {res['final_val_acc']:.4f}")

    return results


if __name__ == "__main__":
    model, base = build_vgg16_model(use_pretrained_weights=True)
    model.summary()
    print("\n✅ VGG16 Transfer Learning model built successfully")
