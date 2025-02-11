# lines 66-186 of the code in this module are from the YouTube Tutorial
# "Build a Generative Adversarial Neural Network with Tensorflow and Python | Deep Learning Projects"
# by Nicholas Renotte, availabe at https://www.youtube.com/watch?v=AALBGpLbj6Q
# the original source code is available at https://github.com/nicknochnack/GANBasics

from music21 import *
import os
from fractions import Fraction
import tensorflow as tf
import numpy as np
import json
import ast
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seq_length = 32
batch_size = 64
vocab_size = 48

def preview_data(image_array, image_num):
    start_index = random.randint(0, len(image_array))
    image = []
    for i in range(seq_length):
        image.append(image_array[start_index][i])
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)
    img = tf.keras.preprocessing.image.array_to_img(image)
    img.save(f'Encoded Images/dataeset_img_{image_num}:{start_index}.png')

def load_data():
    steps = []  # there are 6.2 million steps in total
    count = 0
    with open('GAN Text Data/GAN Data', 'r') as f:
        for line in f:
            count += 1
            if count > 1000000000:
                break
            if count > 0:
                note = ast.literal_eval(line[:-1])
                steps.append(note)
        print(count/32)
    # split sequence of all notes in images
    images = []
    for i in range(int(len(steps)/seq_length)): # i.e number of images
        image = []
        for n in range(seq_length):
            image.append(steps[seq_length*i+n])
        images.append(image)
    images = np.array(images)
    return images

def preprocess_steps(steps):
    # Convert numpy array into a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(steps)
    # Shuffle your dataset
    # Batch your dataset into groups of 64
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(int(tf.data.experimental.cardinality(dataset)))
    #print(tf.data.experimental.cardinality(dataset)) # prints num batches
    dataset = dataset.prefetch(batch_size)
    dataset.as_numpy_iterator().next().shape
    return dataset

def build_generator():
    model = tf.keras.models.Sequential()
    # Takes in random values and reshapes it to 7x7x128
    # Input layer
    model.add(tf.keras.layers.Dense(int(seq_length/4) * int(vocab_size/4) * 128, input_dim=128))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Reshape((int(seq_length/4), int(vocab_size/4), 128)))
    # Upsampling block 1: shape (x, y, z) -> (2x, 2y, z)
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, 5, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Upsampling block 2
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, 5, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Convolutional block 1
    model.add(tf.keras.layers.Conv2D(128, 5, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Convolutional block 2
    model.add(tf.keras.layers.Conv2D(128, 4, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Conv layer to get to one channel
    model.add(tf.keras.layers.Conv2D(1, 4, padding='same', activation='sigmoid'))
    #model.add(tf.keras.layers.Activation(custom_sigmoid))
    model.summary()
    return model

def build_generator2():
    model = tf.keras.models.Sequential()
    # Takes in random values and reshapes it to 7x7x128
    # Input layer
    model.add(tf.keras.layers.Dense(int(seq_length/4) * int(vocab_size/4) * 128, input_dim=128))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Reshape((int(seq_length/4), int(vocab_size/4), 128)))
    # Upsampling block 1: shape (x, y, z) -> (2x, 2y, z)
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, 10, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Upsampling block 2
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, 10, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Convolutional block 1
    model.add(tf.keras.layers.Conv2D(128, 8, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Convolutional block 2
    model.add(tf.keras.layers.Conv2D(128, 6, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    # Conv layer to get to one channel
    model.add(tf.keras.layers.Conv2D(1, 4, padding='same', activation='sigmoid'))
    #model.add(tf.keras.layers.Activation(custom_sigmoid))
    model.summary()
    return model

def build_discriminator(): # output 1 reperesents false
    model = tf.keras.models.Sequential()
    # First Conv Block
    model.add(tf.keras.layers.Conv2D(32, 5, input_shape=(seq_length, vocab_size, 1)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Second Conv Block
    model.add(tf.keras.layers.Conv2D(64, 5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Third Conv Block
    model.add(tf.keras.layers.Conv2D(128, 5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Fourth Conv Block
    model.add(tf.keras.layers.Conv2D(256, 4))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Flatten then pass to dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

def build_discriminator2(): # output 1 reperesents false
    model = tf.keras.models.Sequential()
    # First Conv Block
    model.add(tf.keras.layers.Conv2D(32, 10, input_shape=(seq_length, vocab_size, 1)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Second Conv Block
    model.add(tf.keras.layers.Conv2D(64, 10))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Third Conv Block
    model.add(tf.keras.layers.Conv2D(128, 8))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Fourth Conv Block
    model.add(tf.keras.layers.Conv2D(256, 6))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # Flatten then pass to dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

g_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
d_opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
g_loss = tf.keras.losses.BinaryCrossentropy()
d_loss = tf.keras.losses.BinaryCrossentropy()

class MusicGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class
        super().__init__(*args, **kwargs)
        # Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        # Compile with base class
        super().compile(*args, **kwargs)
        # Create attributes for losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((batch_size, 128, 1)), training=False)
        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
            # Add some noise to the TRUE outputs (labels)
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            # Calculate loss - BINARYCROSS
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
        # Apply backpropagation - nn learn
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((batch_size, 128, 1)), training=True)
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)
            # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

class ModelMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=20, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        vector_seed = np.random.randint(1000)
        vector = tf.random.uniform((self.num_img, self.latent_dim, 1), seed=vector_seed)
        generated_images = self.model.generator(vector)

        #print(generated_images)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(f'Encoded Images/generated_img_{epoch+1}_{i}.png')

for filename in os.listdir('Encoded Images'):
    os.remove(f'Encoded Images/{filename}')
#os.remove('Models/gan1.h5')
#os.remove('Models/d5*.h5')
images_array = load_data()
for i in range(5):
    preview_data(images_array, i+1)
ds = preprocess_steps(images_array)

generator = tf.keras.models.load_model('Models/g34.h5')
discriminator = tf.keras.models.load_model('Models/d34.h5')
gan = MusicGAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss)
hist = gan.fit(ds, epochs=2, callbacks=[ModelMonitor()])
generator.save('Models/g38.h5')
discriminator.save('Models/d38.h5')
imgs = generator.predict(tf.random.normal((16, 128, 1)))
