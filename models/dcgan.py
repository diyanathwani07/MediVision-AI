"""
DCGAN - Deep Convolutional GAN for Medical Image Synthesis
Covers: Experiment 8 - GAN that generates realistic medical-related images
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os


class MedicalDCGAN:
    """
    DCGAN for generating synthetic brain MRI images.
    Trained per class to generate targeted augmentation data.
    """

    def __init__(self, latent_dim=128, img_shape=(64, 64, 1)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

        # Track losses for monitoring
        self.gen_losses = []
        self.disc_losses = []

    def build_generator(self):
        """
        Generator: Latent vector → Synthetic MRI image
        Uses transposed convolutions to upsample from 4x4 to 64x64
        """
        model = models.Sequential(name="Generator")

        # Foundation: 4x4 feature map
        model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Reshape((4, 4, 512)))

        # Upsample to 8x8
        model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Upsample to 16x16
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Upsample to 32x32
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        # Upsample to 64x64
        model.add(layers.Conv2DTranspose(self.img_shape[2], (4, 4), strides=(2, 2),
                                          padding='same', use_bias=False, activation='tanh'))

        return model

    def build_discriminator(self):
        """
        Discriminator: MRI image → Real/Fake probability
        Uses strided convolutions to downsample
        """
        model = models.Sequential(name="Discriminator")

        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                 input_shape=self.img_shape))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))  # No sigmoid — use from_logits=True

        return model

    def build_gan(self):
        """Combine generator + discriminator into GAN."""
        # Freeze discriminator weights during generator training
        self.discriminator.trainable = False

        gan_input = tf.keras.Input(shape=(self.latent_dim,))
        fake_image = self.generator(gan_input)
        validity = self.discriminator(fake_image)

        gan = models.Model(gan_input, validity, name="DCGAN")
        return gan

    @tf.function
    def train_step(self, real_images, batch_size):
        """Single training step with gradient tape."""
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            # WGAN-style loss for stable training
            gen_loss = -tf.reduce_mean(fake_output)
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        # Apply gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset, epochs=200, sample_interval=10, save_dir="gan_outputs/"):
        """Full training loop with periodic image sampling."""
        os.makedirs(save_dir, exist_ok=True)
        fixed_noise = tf.random.normal([16, self.latent_dim])

        print(f"\n🧬 Training DCGAN for {epochs} epochs...")
        print(f"   Latent dim: {self.latent_dim} | Image shape: {self.img_shape}")

        for epoch in range(epochs):
            epoch_gen_loss = []
            epoch_disc_loss = []

            for batch in dataset:
                batch_size = tf.shape(batch)[0]
                g_loss, d_loss = self.train_step(batch, batch_size)
                epoch_gen_loss.append(float(g_loss))
                epoch_disc_loss.append(float(d_loss))

            avg_g = np.mean(epoch_gen_loss)
            avg_d = np.mean(epoch_disc_loss)
            self.gen_losses.append(avg_g)
            self.disc_losses.append(avg_d)

            if epoch % sample_interval == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1:>4}/{epochs}] "
                      f"G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")
                self.save_samples(fixed_noise, epoch, save_dir)

        print("✅ DCGAN training complete!")
        self.generator.save(f"{save_dir}/generator_final.h5")
        return self.gen_losses, self.disc_losses

    def save_samples(self, noise, epoch, save_dir):
        """Generate and save sample images."""
        generated = self.generator(noise, training=False)
        generated = (generated + 1) / 2.0  # [-1,1] → [0,1]

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'GAN Generated MRI — Epoch {epoch}', fontsize=14)

        for i, ax in enumerate(axes.flatten()):
            img = generated[i].numpy()
            if img.shape[-1] == 1:
                ax.imshow(img[:, :, 0], cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch:04d}.png", dpi=100, bbox_inches='tight')
        plt.close()

    def generate_augmentation_batch(self, n_samples=100):
        """
        Generate n_samples synthetic MRI images for dataset augmentation.
        Returns normalized images in [0, 1] range.
        """
        noise = tf.random.normal([n_samples, self.latent_dim])
        synthetic = self.generator(noise, training=False)
        synthetic = (synthetic + 1) / 2.0  # Denormalize
        return synthetic.numpy()

    def plot_training_curves(self):
        """Visualize GAN training dynamics."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.gen_losses, label='Generator Loss', color='#e74c3c')
        ax.plot(self.disc_losses, label='Discriminator Loss', color='#3498db')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DCGAN Training Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("gan_outputs/training_curves.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    # Quick architecture test
    gan = MedicalDCGAN(latent_dim=128, img_shape=(64, 64, 1))
    gan.generator.summary()
    gan.discriminator.summary()
    print("\n✅ DCGAN architecture created successfully")

    # Test forward pass
    test_noise = tf.random.normal([4, 128])
    test_output = gan.generator(test_noise)
    print(f"   Generator output shape: {test_output.shape}")
