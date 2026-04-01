"""
Grad-CAM Visualization — Explainable AI for medical imaging
Shows which regions of the MRI the model focuses on for classification
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image (1, H, W, C)
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Class index (None = uses predicted class)
    
    Returns:
        heatmap: Normalized heatmap array (H, W)
    """
    # Create gradient model: inputs → [last_conv_output, predictions]
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of class score w.r.t. last conv layer output
    grads = tape.gradient(class_channel, last_conv_output)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Returns:
        superimposed: RGB image with heatmap overlay
    """
    # Resize heatmap to image size
    img_h, img_w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)

    superimposed = cv2.addWeighted(original_img, 1 - alpha,
                                    heatmap_colored, alpha, 0)
    return superimposed


def visualize_gradcam_batch(model, images, labels, class_names,
                             last_conv_layer='block5_conv3', n=8):
    """
    Visualize Grad-CAM for a batch of images with side-by-side comparison.
    """
    fig, axes = plt.subplots(n, 3, figsize=(15, n * 4))
    fig.suptitle('Grad-CAM Visualization — Where the Model Looks',
                 fontsize=16, fontweight='bold')

    columns = ['Original MRI', 'Grad-CAM Heatmap', 'Prediction']

    for col_idx, col_name in enumerate(columns):
        axes[0, col_idx].set_title(col_name, fontsize=12, pad=10)

    for i in range(min(n, len(images))):
        img = images[i:i+1]  # (1, H, W, C)
        preds = model.predict(img, verbose=0)
        pred_idx = np.argmax(preds[0])
        true_idx = np.argmax(labels[i]) if len(labels.shape) > 1 else labels[i]
        confidence = preds[0][pred_idx]

        # Compute Grad-CAM
        heatmap = get_gradcam_heatmap(model, img, last_conv_layer, pred_index=pred_idx)

        # Denormalize image for display
        display_img = ((img[0].numpy() + 1) / 2.0 * 255).astype(np.uint8)

        # Overlay
        overlay = overlay_gradcam(display_img, heatmap)

        # Plot original
        axes[i, 0].imshow(display_img)
        axes[i, 0].set_ylabel(f"True: {class_names[true_idx]}", fontsize=9)
        axes[i, 0].axis('off')

        # Plot heatmap
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].axis('off')

        # Plot overlay with prediction
        correct = pred_idx == true_idx
        color = '#2ecc71' if correct else '#e74c3c'
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_xlabel(
            f"Pred: {class_names[pred_idx]} ({confidence:.1%})",
            color=color, fontsize=9
        )
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/gradcam_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Grad-CAM saved to outputs/gradcam_visualization.png")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot styled confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=13)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=13)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("✅ Visualization module loaded")
    print("   Functions: get_gradcam_heatmap, overlay_gradcam,")
    print("              visualize_gradcam_batch, plot_confusion_matrix")
