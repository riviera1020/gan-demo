import os
import sys
import numpy as np
from PIL import Image
from glob import glob

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

class WGAN():
    def __init__(self, inputs_dir):
        self.inputs_dir = inputs_dir
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
        """Calculates the gradient penalty loss for a batch of "averaged" samples.

        In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
        that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
        this function at all points in the input space. The compromise used in the paper is to choose random points
        on the lines between real and generated samples, and check the gradients at these points. Note that it is the
        gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

        In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
        Then we get the gradients of the discriminator w.r.t. the input averaged samples.
        The l2 norm and penalty can then be calculated for this gradient.

        Note that this loss function requires the original averaged samples as input, but Keras only supports passing
        y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
        averaged_samples argument, and use that for model training."""
        gradients = K.gradients(K.sum(y_pred), averaged_samples)
        gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        return gradient_penalty


    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation="relu", input_shape=noise_shape))
        model.add(Reshape((16, 16, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.summary()

        img = Input(shape=img_shape)
        features = model(img)
        valid = Dense(1, activation="linear")(features)

        return Model(img, valid)

    def get_image(self, image_path, width, height, mode):

        image = Image.open(image_path)
        # image = image.resize([width, height], Image.BILINEAR)
        if image.size != (width, height):
        # Remove most pixels that aren't part of a face
            face_width = face_height = 108
            j = (image.size[0] - face_width) // 2
            i = (image.size[1] - face_height) // 2
            image = image.crop([j, i, j + face_width, i + face_height])
            image = image.resize([width, height])
        return np.array(image.convert(mode))

    def get_batch(self, image_files, width, height, mode):
        data_batch = np.array(
            [self.get_image(sample_file, width, height, mode) for sample_file in image_files])

        return data_batch

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = self.get_batch(glob(os.path.join(self.inputs_dir, '*.jpg'))[:5000], 64, 64, 'RGB')

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                noise = np.random.normal(0, 1, (half_batch, 100))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, -np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, -np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        plt.show()
        #fig.savefig("output/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':

    inputs_dir = './one_tag_images'
    wgan = WGAN(inputs_dir)
    wgan.train(epochs=4000, batch_size=32, save_interval=50)

