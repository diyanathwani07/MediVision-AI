"""
Unit tests for MediVision AI model architectures
Run: pytest tests/ -v
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tensorflow as tf


class TestCustomCNN:
    def test_build_model(self):
        from models.custom_cnn import build_custom_cnn
        model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=4)
        assert model is not None
        assert model.output_shape == (None, 4)

    def test_forward_pass(self):
        from models.custom_cnn import build_custom_cnn
        model = build_custom_cnn()
        dummy = tf.random.normal([2, 224, 224, 3])
        out = model(dummy, training=False)
        assert out.shape == (2, 4)
        # Softmax sums to 1
        assert np.allclose(out.numpy().sum(axis=1), 1.0, atol=1e-5)

    def test_output_probabilities(self):
        from models.custom_cnn import build_custom_cnn
        model = build_custom_cnn()
        dummy = tf.random.normal([1, 224, 224, 3])
        out = model(dummy, training=False).numpy()[0]
        assert all(p >= 0 for p in out)
        assert all(p <= 1 for p in out)


class TestVGG16Transfer:
    def test_build_pretrained(self):
        from models.vgg16_transfer import build_vgg16_model
        model, base = build_vgg16_model(use_pretrained_weights=True, num_classes=4)
        assert model.output_shape == (None, 4)

    def test_build_from_scratch(self):
        from models.vgg16_transfer import build_vgg16_model
        model, base = build_vgg16_model(use_pretrained_weights=False, num_classes=4)
        assert model.output_shape == (None, 4)

    def test_frozen_layers(self):
        from models.vgg16_transfer import build_vgg16_model
        model, base = build_vgg16_model(use_pretrained_weights=True, freeze_layers=15)
        frozen = sum(1 for l in base.layers if not l.trainable)
        assert frozen == 15


class TestDCGAN:
    def test_generator_shape(self):
        from models.dcgan import MedicalDCGAN
        gan = MedicalDCGAN(latent_dim=64, img_shape=(64, 64, 1))
        noise = tf.random.normal([4, 64])
        out = gan.generator(noise, training=False)
        assert out.shape == (4, 64, 64, 1)

    def test_discriminator_shape(self):
        from models.dcgan import MedicalDCGAN
        gan = MedicalDCGAN(latent_dim=64, img_shape=(64, 64, 1))
        fake_imgs = tf.random.normal([4, 64, 64, 1])
        out = gan.discriminator(fake_imgs, training=False)
        assert out.shape == (4, 1)

    def test_generate_augmentation(self):
        from models.dcgan import MedicalDCGAN
        gan = MedicalDCGAN(latent_dim=64, img_shape=(64, 64, 1))
        batch = gan.generate_augmentation_batch(n_samples=10)
        assert batch.shape == (10, 64, 64, 1)
        assert batch.min() >= 0 and batch.max() <= 1


class TestDataLoader:
    def test_preprocess_range(self):
        from utils.data_loader import preprocess
        dummy = tf.constant(np.random.randint(0, 255, (224, 224, 3)), dtype=tf.float32)
        processed = preprocess(dummy, augment=False)
        assert float(processed.numpy().min()) >= -1.0
        assert float(processed.numpy().max()) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
