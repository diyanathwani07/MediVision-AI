"""
Flask Web Application — MediVision AI Dashboard
Real-time brain tumor classification via REST API + Web UI
Covers: Experiment 10 - Real-Time Image Classification
"""

import os
import sys
import json
import time
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import prepare_image_for_inference, CLASSES
from utils.visualization import get_gradcam_heatmap, overlay_gradcam

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_INFO = {
    'glioma':      {'label': 'Glioma', 'color': '#e74c3c', 'risk': 'High'},
    'meningioma':  {'label': 'Meningioma', 'color': '#f39c12', 'risk': 'Medium'},
    'notumor':     {'label': 'No Tumor', 'color': '#2ecc71', 'risk': 'None'},
    'pituitary':   {'label': 'Pituitary Tumor', 'color': '#3498db', 'risk': 'Medium'}
}

# ── Model Loading ─────────────────────────────────────────────────────────────
models_loaded = {}

def load_models():
    """Load trained models at startup."""
    global models_loaded
    print("\n🔄 Loading MediVision AI models...")

    model_paths = {
        'cnn':   'checkpoints/cnn_best.h5',
        'vgg16': 'checkpoints/vgg16_finetuned_best.h5',
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            models_loaded[name] = tf.keras.models.load_model(path)
            print(f"   ✅ {name.upper()} loaded from {path}")
        else:
            print(f"   ⚠️  {name.upper()} not found at {path} — train first")

    if not models_loaded:
        print("   ℹ️  No trained models found. API will return demo predictions.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body: multipart/form-data with 'image' file + optional 'model' param
    Returns: JSON with predictions, confidence, Grad-CAM, inference time
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    model_name = request.form.get('model', 'vgg16')

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded file
    filename = f"upload_{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        start_time = time.time()

        # Preprocess
        img_tensor = prepare_image_for_inference(filepath)

        # Inference
        if model_name in models_loaded:
            model = models_loaded[model_name]
            predictions = model.predict(img_tensor, verbose=0)[0]
        else:
            # Demo mode — return mock predictions for testing UI
            predictions = np.random.dirichlet(np.ones(4) * 2)

        inference_time = (time.time() - start_time) * 1000  # ms

        pred_idx = int(np.argmax(predictions))
        pred_class = CLASSES[pred_idx]
        confidence = float(predictions[pred_idx])

        # Build Grad-CAM (only for loaded models)
        gradcam_b64 = None
        if model_name in models_loaded:
            try:
                last_conv = 'block5_conv3' if model_name == 'vgg16' else 'conv2d_3'
                heatmap = get_gradcam_heatmap(models_loaded[model_name],
                                               img_tensor, last_conv)

                # Overlay on original image
                orig_img = np.array(Image.open(filepath).resize((224, 224)))
                if orig_img.ndim == 2:
                    orig_img = np.stack([orig_img]*3, axis=-1)
                overlay_img = overlay_gradcam(orig_img, heatmap)

                # Encode as base64
                pil_img = Image.fromarray(overlay_img.astype(np.uint8))
                buf = BytesIO()
                pil_img.save(buf, format='PNG')
                gradcam_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Grad-CAM error: {e}")

        response = {
            'success': True,
            'prediction': {
                'class': pred_class,
                'label': CLASS_INFO[pred_class]['label'],
                'confidence': round(confidence * 100, 2),
                'risk_level': CLASS_INFO[pred_class]['risk'],
                'color': CLASS_INFO[pred_class]['color']
            },
            'all_probabilities': {
                CLASSES[i]: round(float(predictions[i]) * 100, 2)
                for i in range(len(CLASSES))
            },
            'model_used': model_name,
            'inference_time_ms': round(inference_time, 1),
            'gradcam': gradcam_b64,
            'image_path': filepath
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """List available models and their status."""
    return jsonify({
        'models': [
            {'id': 'cnn', 'name': 'Custom CNN', 'loaded': 'cnn' in models_loaded},
            {'id': 'vgg16', 'name': 'VGG16 Transfer', 'loaded': 'vgg16' in models_loaded}
        ],
        'classes': CLASSES,
        'class_info': CLASS_INFO
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models_loaded.keys()),
        'tensorflow_version': tf.__version__
    })


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_models()
    print("\n🚀 MediVision AI Dashboard running at http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
