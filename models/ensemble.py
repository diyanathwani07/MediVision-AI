"""
Ensemble Model — Combines CNN + VGG16 for improved accuracy
Covers: Experiment 9 - Ensemble learning techniques
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score


CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']


class EnsembleClassifier:
    """
    Soft-voting ensemble of CNN and VGG16 models.
    Weights optimized on validation set.
    """

    def __init__(self, models: dict, weights: dict = None):
        """
        Args:
            models: {'cnn': model1, 'vgg16': model2}
            weights: {'cnn': 0.4, 'vgg16': 0.6} — defaults to equal weighting
        """
        self.models = models
        if weights is None:
            n = len(models)
            self.weights = {name: 1.0 / n for name in models}
        else:
            self.weights = weights

    def predict_proba(self, img_tensor):
        """Weighted average of softmax outputs from all models."""
        ensemble_probs = np.zeros(len(CLASSES))

        for name, model in self.models.items():
            probs = model.predict(img_tensor, verbose=0)[0]
            ensemble_probs += self.weights[name] * probs

        return ensemble_probs

    def predict(self, img_tensor):
        """Return predicted class index."""
        probs = self.predict_proba(img_tensor)
        return int(np.argmax(probs))

    def evaluate(self, test_dataset):
        """Evaluate ensemble on test set with full classification report."""
        y_true, y_pred = [], []

        print("\n🔍 Evaluating Ensemble on test set...")
        for images, labels in test_dataset:
            for i in range(len(images)):
                img = images[i:i+1]
                true_idx = int(np.argmax(labels[i]))
                pred_idx = self.predict(img)
                y_true.append(true_idx)
                y_pred.append(pred_idx)

        acc = accuracy_score(y_true, y_pred)
        print(f"\n✅ Ensemble Test Accuracy: {acc:.4f}")
        print("\n" + classification_report(y_true, y_pred, target_names=CLASSES))

        return y_true, y_pred

    def optimize_weights(self, val_dataset, search_steps=10):
        """
        Grid search for optimal ensemble weights on validation set.
        """
        print("\n⚙️  Optimizing ensemble weights...")
        best_acc, best_weights = 0, self.weights.copy()

        for alpha in np.linspace(0.1, 0.9, search_steps):
            self.weights = {'cnn': alpha, 'vgg16': 1 - alpha}

            y_true, y_pred = [], []
            for images, labels in val_dataset:
                for i in range(len(images)):
                    y_true.append(int(np.argmax(labels[i])))
                    y_pred.append(self.predict(images[i:i+1]))

            acc = accuracy_score(y_true, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_weights = self.weights.copy()

        self.weights = best_weights
        print(f"   Best weights: CNN={best_weights['cnn']:.2f}, "
              f"VGG16={best_weights['vgg16']:.2f}")
        print(f"   Best val accuracy: {best_acc:.4f}")
        return best_weights


def load_ensemble(cnn_path="checkpoints/cnn_best.h5",
                  vgg16_path="checkpoints/vgg16_finetuned_best.h5"):
    """Load both models and return ensemble."""
    models = {}

    if tf.io.gfile.exists(cnn_path):
        models['cnn'] = tf.keras.models.load_model(cnn_path)
        print(f"✅ CNN loaded from {cnn_path}")
    else:
        print(f"⚠️  CNN not found at {cnn_path}")

    if tf.io.gfile.exists(vgg16_path):
        models['vgg16'] = tf.keras.models.load_model(vgg16_path)
        print(f"✅ VGG16 loaded from {vgg16_path}")
    else:
        print(f"⚠️  VGG16 not found at {vgg16_path}")

    if len(models) < 2:
        print("⚠️  Need both models for ensemble. Using available models only.")

    return EnsembleClassifier(models)


if __name__ == "__main__":
    print("✅ Ensemble module ready")
    print("   Usage: ensemble = load_ensemble()")
    print("          probs = ensemble.predict_proba(img_tensor)")
