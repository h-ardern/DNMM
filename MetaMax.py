import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import pickle


class MetaMax:
    def __init__(self, num_classes, batch_size=2256):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.enc = OneHotEncoder(sparse_output=False)
        self.weibull_models = []

    def fit_weibull(self, data, pbar):
        pbar.write("Starting Weibull fitting on CPU with NumPy...")

        # Convert data to NumPy array
        data = data.numpy()
        data = data.flatten()

        # Normalize data
        data_min = np.min(data)
        data_max = np.max(data)
        data = (data - data_min) / (data_max - data_min)

        # Use SciPy to fit the Weibull distribution
        from scipy.stats import weibull_min
        try:
            c, loc, scale = weibull_min.fit(data)
            return np.array([c, loc, scale], dtype=np.float32)
        except Exception as e:
            return np.array([1.0, 0.0, 1.0], dtype=np.float32)

    def fit(self, features, labels):
        print("Starting fit method on CPU with NumPy...")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")

        try:
            labels = self.enc.fit_transform(labels.reshape(-1, 1))
            print(f"Labels after encoding shape: {labels.shape}")

            self.weibull_models = []

            pbar = tqdm(range(self.num_classes), desc="Fitting Weibull models for classes")
            for i in pbar:
                pbar.write(f"Fitting Weibull for class {i}...")

                class_indices = np.where(labels[:, i] == 1)[0]
                num_samples = len(class_indices)
                pbar.write(f"Class {i} features shape: {num_samples}")

                if num_samples < 10:
                    pbar.write(f"Not enough samples for class {i}. Using default Weibull parameters.")
                    self.weibull_models.append(np.array([1.0, 0.0, 1.0], dtype=np.float32))
                    continue

                weibull_model_params = []
                batch_pbar = tqdm(range(0, num_samples, self.batch_size), desc=f"Processing class {i} batches", leave=False)
                for start in batch_pbar:
                    end = min(start + self.batch_size, num_samples)
                    batch_indices = class_indices[start:end]
                    batch_features = features[batch_indices]
                    batch_features_tensor = tf.convert_to_tensor(batch_features, dtype=tf.float32)
                    pbar.write(f"Batch features tensor shape: {batch_features_tensor.shape}")

                    # Flatten the batch features tensor to 1D
                    batch_features_flat = tf.reshape(batch_features_tensor, [-1])
                    pbar.write(f"Flattened batch features tensor shape: {batch_features_flat.shape}")

                    # Convert to NumPy array
                    weibull_model = self.fit_weibull(batch_features_flat, pbar)
                    weibull_model_params.append(weibull_model)

                weibull_model_params = np.stack(weibull_model_params)
                self.weibull_models.append(np.mean(weibull_model_params, axis=0))
            pbar.close()
        except Exception as e:
            print(f"Error in fit method: {e}")

    def calibrate(self, logits):
        print("Starting calibration on CPU with NumPy...")

        calibrated_scores = np.zeros_like(logits)

        for i in tqdm(range(self.num_classes), desc="Calibrating logits"):
            weibull_model = self.weibull_models[i]
            c, loc, scale = weibull_model
            dist = tfp.distributions.Weibull(c, scale)
            logits_tensor = tf.convert_to_tensor(logits[:, i] - loc, dtype=tf.float32)
            calibrated_scores[:, i] = dist.cdf(logits_tensor).numpy()

        print("Calibration completed.")
        return calibrated_scores

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)