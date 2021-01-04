## From Data
import pandas as pd
import numpy as np
## ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
## Web Scraping 
import requests
from bs4 import BeautifulSoup
## Image manipulation
from PIL import Image
from torchvision import transforms 
import matplotlib.pyplot as plt
## Filing Functionality 
import io
import os
import hashlib



im = Image.open("C:/Users/Admin/Desktop/ART/data/Abstract_gallery/Abstract_image_50.jpg")
#im.show()
width, height = im.size
size = 100, 100


trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

tensorPic = trans1(im)
#print(tensorPic.shape)
tensorPic[2] = 0.75
#print(tensorPic)
#trans(tensorPic[1]).show()

#resize images to 100x100
im.thumbnail(size, Image.ANTIALIAS)
#im.show()

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Build the encoder
"""

latent_dim = 2
#https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc#:~:text=The%20input%20shape&text=In%20Keras%2C%20the%20input%20layer,shape%20as%20your%20training%20data.
# 30 images of 100x100 pixels in RGB (3 channels)
encoder_inputs = keras.Input(shape=(30,100,100,3))
##2D only moves the filter in X and Y 
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#encoder.summary()

"""
## Build the decoder
"""
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")



class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
print(np.concatenate([x_train, x_test]).shape)
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#vae = VAE(encoder, decoder)
#vae.compile(optimizer=keras.optimizers.Adam())
#vae.fit(mnist_digits, epochs=30, batch_size=128)



















#plt.imshow(trans(trans1(im)))


#tf.reset_default_graph()

#batch_size = 64

#X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
#Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
#Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
#keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

#dec_in_channels = 1
#n_latent = 8

#reshaped_dim = [-1, 7, 7, dec_in_channels]
#inputs_decoder = 49 * dec_in_channels / 2


#def lrelu(x, alpha=0.3):
#    return tf.maximum(x, tf.multiply(x, alpha))
































def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

#persist_image(r"C:\Users\Admin\Desktop\ART", "https://d3d00swyhr67nd.cloudfront.net/w1200h1200/collection/SFK/SED/SFK_SED_ST_1992_9_587-001.jpg")

#artLoc = pd.read_csv("./painting_dataset_2018.csv")
#print(artLoc.describe())

#print(artLoc['Image URL'][0])
#result = requests.get(artLoc['Image URL'][0])