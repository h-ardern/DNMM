import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pickle
class KLDivergenceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        self.add_loss(kl_loss)
        return inputs

class GeMOS:
    def __init__(self, input_dim, latent_dim, num_components, batch_size, cache_dir='/tmp/cache'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.vae = self._build_vae_model()
        self.gmm_models = []

    def _build_vae_model(self):
        # Define the encoder
        inputs = layers.Input(shape=(self.input_dim,))
        h = layers.Dense(512, activation='relu')(inputs)
        z_mean = layers.Dense(self.latent_dim)(h)
        z_log_var = layers.Dense(self.latent_dim)(h)

        z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])

        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim), dtype=tf.float16)
            return tf.cast(z_mean, tf.float16) + tf.exp(0.5 * tf.cast(z_log_var, tf.float16)) * epsilon

        z = layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # Define the decoder
        decoder_h = layers.Dense(512, activation='relu')
        decoder_mean = layers.Dense(self.input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        outputs = decoder_mean(h_decoded)

        # Instantiate the VAE model
        vae = Model(inputs, outputs)
        vae.compile(optimizer='adam', loss='mse')
        return vae

    def fit(self, X, labels):
        print("Fitting GeMOS...")
        print(f"Shape of train_features before reshaping: {X.shape}")
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        print(f"Shape of train_features after reshaping: {X.shape}")

        dataset = self.create_and_cache_dataset(X)
        self.vae.fit(dataset, epochs=50, verbose=2)

    def create_and_cache_dataset(self, X):
        dataset = tf.data.Dataset.from_tensor_slices((X, X)).batch(self.batch_size)
        dataset = dataset.take(len(X)).cache(self.cache_dir).repeat().batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def predict(self, X):
        print("Computing GeMOS scores...")
        scores = np.zeros((X.shape[0], len(self.gmm_models) + 1))
        reconstructions = self.vae.predict(X)

        for i, gmm in enumerate(self.gmm_models):
            log_prob = gmm.score_samples(reconstructions)
            scores[:, i] = log_prob

        scores[:, -1] = np.max(scores[:, :-1], axis=1) - np.min(scores[:, :-1], axis=1)
        return scores

    def detect_unknown_classes(self, X, threshold=0.5):
        print("Detecting unknown classes...")
        scores = self.predict(X)
        unknown_class_predictions = scores[:, -1] > threshold

        num_unknown_predictions = np.sum(unknown_class_predictions)
        print(f"Number of unknown class predictions: {num_unknown_predictions}")

        return unknown_class_predictions, scores

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"GeMOS model saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            print(f"Loading GeMOS model from {file_path}")
            return pickle.load(file)
