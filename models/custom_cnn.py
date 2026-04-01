"""
Custom CNN Model for Brain Tumor Classification
Covers: Experiment 4 - CNN with hyperparameter tuning (Experiment 5)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras_tuner as kt
import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.5):
    """
    Build a custom CNN architecture from scratch.
    
    Architecture:
        - 4 Convolutional blocks with BatchNorm + MaxPooling
        - Global Average Pooling
        - Dense classifier with Dropout regularization
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate / 2),
        layers.Dense(num_classes, activation='softmax')
    ], name="MediVision_CNN")

    return model


def build_tunable_cnn(hp):
    """
    Keras Tuner compatible model builder for hyperparameter search.
    Covers: Experiment 5 - CNN with different hyperparameters
    """
    model = models.Sequential(name="Tunable_CNN")

    # Tune number of conv blocks
    num_blocks = hp.Int('num_blocks', min_value=2, max_value=4, step=1)

    input_added = False
    for i in range(num_blocks):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32)
        
        if not input_added:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu',
                                    padding='same', input_shape=(224, 224, 3)))
            input_added = True
        else:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))

        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.GlobalAveragePooling2D())

    # Tune dense layers
    dense_units = hp.Int('dense_units', min_value=128, max_value=1024, step=128)
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))

    # Tune optimizer
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    optimizers = {
        'adam': tf.keras.optimizers.Adam(learning_rate=lr),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=lr),
        'sgd': tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    }

    model.compile(
        optimizer=optimizers[optimizer_choice],
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def run_hyperparameter_search(train_data, val_data, max_trials=20):
    """
    Run Keras Tuner Bayesian optimization for best architecture.
    """
    tuner = kt.BayesianOptimization(
        build_tunable_cnn,
        objective='val_accuracy',
        max_trials=max_trials,
        directory='kt_results',
        project_name='medivision_tuning'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_data, validation_data=val_data,
                 epochs=30, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nBest Hyperparameters:")
    for key, val in best_hps.values.items():
        print(f"  {key}: {val}")

    return tuner.get_best_models(num_models=1)[0], best_hps


if __name__ == "__main__":
    # Quick test
    model = build_custom_cnn()
    model.summary()
    print("\n✅ Custom CNN built successfully")
