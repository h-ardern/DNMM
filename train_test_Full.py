import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.ops.gen_batch_ops import batch
from tqdm import tqdm
from MetaMax import MetaMax
from DenseNet import DenseNet
from DataStripper import remove_class_and_save
# from GeMOS import GeMOS -- From Failed Generative implementation caused by memory constraints
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import *
import datetime

#Function to print ASCII banner
def print_ascii_banner(file_path):
    try:
        with open(file_path, 'r') as f:
            banner = f.read()
            print(banner)
    except Exception as e:
        print(f"Error loading banner: {e}")

# Print the ASCII banner from the text file
banner_file_path = 'Banner.txt'
print_ascii_banner(banner_file_path)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Enable mixed precision
set_global_policy('mixed_float16')

# List Available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define Batch Size
Batch_size = 4 

# Paths to datasets
filtered_file = 'iot23_combined_filtered.csv'
input_file = 'iot23_combined_new.csv'

# Load the filtered dataset
if os.path.exists(filtered_file):
    print("Loading Stripped Dataset")
    df_filtered = pd.read_csv(filtered_file)
    print("Dataset Loaded !")
else:
    input_file = 'iot23_combined_new.csv'
    output_file = filtered_file
    class_to_remove = 'C&C'  # Change this to the class you want to remove, In results discussed in paper C&C was class removed.
    remove_class_and_save(input_file, output_file, class_to_remove)
    df_filtered = pd.read_csv(filtered_file)

# Load the full dataset
df_full = pd.read_csv(input_file)
df_full = df_full.drop(df_full.columns[0], axis=1)

# Display the first few rows to understand the data structure
print("Filtered dataset:")
print(df_filtered.head())
print("Full dataset:")
print(df_full.head())

# Identify features and target column
features_filtered = df_filtered.drop(columns=['label'])
labels_filtered = df_filtered['label']
features_full = df_full.drop(columns=['label'])
labels_full = df_full['label']

# Convert all object columns to string type to avoid mixed types
categorical_columns_filtered = features_filtered.select_dtypes(include=['object']).columns
categorical_columns_full = features_full.select_dtypes(include=['object']).columns

for column in tqdm(categorical_columns_filtered, desc="Converting categorical columns to strings (filtered)"):
    features_filtered[column] = features_filtered[column].astype(str)

for column in tqdm(categorical_columns_full, desc="Converting categorical columns to strings (full)"):
    features_full[column] = features_full[column].astype(str)

# Encode categorical features with Label Encoding
label_encoder_filtered = LabelEncoder()
label_encoder_full = LabelEncoder()

for column in tqdm(categorical_columns_filtered, desc="Label encoding categorical columns (filtered)"):
    features_filtered[column] = label_encoder_filtered.fit_transform(features_filtered[column])

for column in tqdm(categorical_columns_full, desc="Label encoding categorical columns (full)"):
    features_full[column] = label_encoder_full.fit_transform(features_full[column])

# Encode labels
print("Encoding labels (filtered)...")
labels_encoded_filtered = label_encoder_filtered.fit_transform(labels_filtered)
print("Labels encoded (filtered).")

print("Encoding labels (full)...")
labels_encoded_full = label_encoder_full.fit_transform(labels_full)
print("Labels encoded (full).")

# Standardize the feature columns
print("Standardizing features (filtered)...")
scaler_filtered = StandardScaler()
features_scaled_filtered = scaler_filtered.fit_transform(features_filtered)
print("Features standardized (filtered).")

print("Standardizing features (full)...")
scaler_full = StandardScaler()
features_scaled_full = scaler_full.fit_transform(features_full)
print("Features standardized (full).")

# Split into train and test sets (filtered dataset)
print("Splitting data into train and test sets (filtered)...")
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
    features_scaled_filtered, labels_encoded_filtered, test_size=0.2, random_state=42)
print("Data split into train and test sets (filtered).")

# Convert labels to categorical
print("Converting labels to categorical (filtered)...")
num_classes_filtered = len(np.unique(labels_encoded_filtered))
y_train_cat_filtered = to_categorical(y_train_filtered, num_classes_filtered)
y_test_cat_filtered = to_categorical(y_test_filtered, num_classes_filtered)
print("Labels converted to categorical (filtered).")

# Calculate new dimensions for reshaping
print("Calculating new dimensions for reshaping (filtered)")
X_train_reshaped_filtered = X_train_filtered.reshape(-1, 20, 1, 1)
X_test_reshaped_filtered = X_test_filtered.reshape(-1, 20, 1, 1)
print(f"Reshaped X_train shape (filtered): {X_train_reshaped_filtered.shape}")
print(f"Reshaped X_test shape (filtered): {X_test_reshaped_filtered.shape}")
print("Done")

# Define the model save path
model_save_path = 'densenet_model.keras'

# TensorBoard log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Check if the model already exists
if os.path.exists(model_save_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_save_path)
    print("Model loaded.")
else:
    # Initialize DenseNet with regularization
    print("Initializing DenseNet")
    input_shape = (20, 1, 1)
    densenet = DenseNet(input_shape=input_shape, num_classes=num_classes_filtered, weight_decay=1e-4)
    model = densenet.get_model()
    model.summary()
    print("Done")

    # Compile the model with a small learning rate
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    # Define ModelCheckpoint and TensorBoard callbacks
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train DenseNet
    print("Training model...")
    history = model.fit(X_train_reshaped_filtered, y_train_cat_filtered, epochs=10, batch_size=Batch_size, verbose=1,
                        callbacks=[early_stopping, checkpoint, tensorboard_callback], validation_split=0.2)
    print("Model trained.")

# Custom function to load npy files with memory mapping
def load_npy_with_mmap(file_path, desc="Loading"):
    print(f"{desc}...")
    return np.load(file_path, mmap_mode='r')

# Paths for saving features
train_features_path = 'train_features.npy'
test_features_path = 'test_features.npy'

train_features = None
test_features = None

# Check if train features are already saved to disk
if os.path.exists(train_features_path):
    print("Loading extracted train features from disk...")
    train_features = load_npy_with_mmap(train_features_path, desc="Loading train features")
    print("Train features loaded from disk.")
else:
    # Extract train features
    print("Extracting train features from penultimate layer using GPU...")
    model_feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    train_features = []

    # Extract features in batches to avoid memory issues
    print("Processing training features in batches...")
    pbar = tqdm(total=len(X_train_reshaped_filtered), desc="Extracting train features", ncols=100)
    for i in range(0, len(X_train_reshaped_filtered), Batch_size):
        end = min(i + Batch_size, len(X_train_reshaped_filtered))
        batch_features = model_feature_extractor.predict(X_train_reshaped_filtered[i:end], verbose=0)
        train_features.append(batch_features)
        pbar.update(end - i)
    pbar.close()
    train_features = np.vstack(train_features)

    # Save extracted train features to disk
    np.save(train_features_path, train_features)
    print("Train features saved to disk.")

# Check if test features are already saved to disk
if os.path.exists(test_features_path):
    print("Loading extracted test features from disk...")
    test_features = load_npy_with_mmap(test_features_path, desc="Loading test features")
    print("Test features loaded from disk.")
else:
    # Extract test features
    print("Extracting test features from penultimate layer using GPU...")
    model_feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    test_features = []

    # Extract features in batches to avoid memory issues
    print("Processing test features in batches...")
    pbar = tqdm(total=len(X_test_reshaped_filtered), desc="Extracting test features", ncols=100)
    for i in range(0, len(X_test_reshaped_filtered), Batch_size):
        end = min(i + Batch_size, len(X_test_reshaped_filtered))
        batch_features = model_feature_extractor.predict(X_test_reshaped_filtered[i:end], verbose=0)
        test_features.append(batch_features)
        pbar.update(end - i)
    pbar.close()
    test_features = np.vstack(test_features)

    # Save extracted test features to disk
    np.save(test_features_path, test_features)
    print("Test features saved to disk.")

print("Features extracted.")

# Evaluate DenseNet model before MetaMax
print("Evaluating DenseNet model...")
logits = model.predict(X_test_reshaped_filtered, batch_size=Batch_size, verbose=1)
predicted = np.argmax(logits, axis=1)
true_labels = np.argmax(y_test_cat_filtered, axis=1)
log_loss_value = log_loss(y_test_cat_filtered, logits)
class_report = classification_report(true_labels, predicted)


# Calculate accuracy, F1 score, and confusion matrix
accuracy = accuracy_score(true_labels, predicted)
f1 = f1_score(true_labels, predicted, average='weighted')
conf_matrix = confusion_matrix(true_labels, predicted ,labels=np.arange(num_classes_filtered))
precision = precision_score(true_labels, predicted, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted, average='weighted', zero_division=1)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Log Loss: {log_loss_value}')
print('Confusion Matrix:')
print(conf_matrix)
print(f'Classification Report: \n{class_report}')

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder_filtered.classes_,
            yticklabels=label_encoder_filtered.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for DenseNet Model')
plt.savefig('DenseNet_Confusion_matrix.png')

# Fit MetaMax and save it
meta_max_file = 'meta_max.pkl'
if not os.path.exists(meta_max_file):
    print("Fitting MetaMax...")
    meta_max = MetaMax(num_classes=num_classes_filtered)
    meta_max.fit(train_features, np.argmax(y_train_cat_filtered, axis=1))
    print("MetaMax fitted.")
    meta_max.save(meta_max_file)
    print(f"MetaMax saved to {meta_max_file}")
else:
    print(f"Loading MetaMax from {meta_max_file}...")
    meta_max = MetaMax.load(meta_max_file)
    print("MetaMax loaded.")

'''
# Archived GeMOS implementation
# Implementing and saving GeMOS
gemos_file = 'gemos_model.pkl'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if not os.path.exists(gemos_file):
    print("Fitting GeMOS...")
    # Define input dimensions and latent dimensions for GeMOS
    input_dim = X_train_filtered.shape[1]  # Replace with the correct input dimension
    latent_dim = 64  # Choose an appropriate latent dimension based on your dataset
    gemos = GeMOS(input_dim=input_dim, latent_dim=latent_dim, num_components=8 ,batch_size=1)
    print(f"Shape of train_features before reshaping: {train_features.shape}")
    reshaped_features = train_features.reshape((train_features.shape[0],-1))
    print(f"Shape of train_features after reshaping: {reshaped_features.shape}")
    # Adjust num_components as needed
    gemos.fit(reshaped_features, np.argmax(y_train_cat_filtered, axis=1))
    print("GeMOS fitted.")
    gemos.save(gemos_file)
    print(f"GeMOS saved to {gemos_file}")
else:
    print(f"Loading GeMOS from {gemos_file}...")
    gemos = GeMOS.load(gemos_file)
    print("GeMOS loaded.")
'''
# Standardize the full dataset features using the previously fitted scaler
features_scaled_full = scaler_filtered.transform(features_full)
X_full_reshaped = features_scaled_full.reshape(-1, 20, 1, 1)

# Paths for saving features
features_full_path = 'features_full.npy'

# Extract features for the full dataset using the DenseNet model
print("Extracting features from the full dataset using GPU...")

# Check if features are already saved to disk
if os.path.exists(features_full_path):
    print("Loading extracted features from disk...")
    features_full_extracted = np.load(features_full_path, mmap_mode='r')
    print("Features loaded from disk.")

else:
    # Collect features from penultimate layer for MetaMax fitting
    print("Extracting features from penultimate layer using GPU...")
    model_feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

    # Create a file to save the features
    with open(features_full_path, 'wb') as f:
        pass

    print("Processing full dataset features in batches...")
    pbar = tqdm(total=len(features_scaled_full), desc="Extracting full dataset features", ncols=100)
    for i in range(0, len(features_scaled_full), Batch_size):
        end = min(i + Batch_size, len(features_scaled_full))
        batch_features = model_feature_extractor.predict(features_scaled_full[i:end].reshape(-1, 20, 1, 1), verbose=0)

        # Append to the file
        with open(features_full_path, 'ab') as f:
            np.save(f, batch_features)

        pbar.update(end - i)
    pbar.close()
    print("Features saved to disk.")

print("Features extracted.")

print('Predicting logits fot the full dataset using Densenet')
# Predict logits for the full dataset using the DenseNet model
logits_full = model.predict(X_full_reshaped, batch_size=Batch_size, verbose=1)

# Define true_labels_full based on the known classes
true_labels_full = labels_encoded_full

# Inference and evaluation with progress bar on the full dataset
print("Evaluating MetaMax on the full dataset...")

print('Calibrating Logits using MetaMax')
# Calibrate the logits using MetaMax
calibrated_scores = meta_max.calibrate(logits_full)

# Detect unknown class predictions
print('Detecting unknown classes using MetaMax')
unknown_class_threshold = 0.85 # Adjust this threshold as needed
unknown_class_predictions = np.max(calibrated_scores, axis=1) < unknown_class_threshold

# Count how many are predicted as unknown
num_unknown_predictions = np.sum(unknown_class_predictions)
print(f"Number of unknown class predictions: {num_unknown_predictions}")

# Filter out the predicted unknown class scores
unknown_class_scores = calibrated_scores[unknown_class_predictions]

# Display the top 10 highest probabilities among the unknown classes
top_unknown_class_indices = np.argsort(-np.max(unknown_class_scores, axis=1))[:10]
print("Top 10 detected unknown class probabilities:")
for idx in top_unknown_class_indices:
    print(f"Sample Index: {idx}, Probabilities: {unknown_class_scores[idx]}")

# Add visualization for unknown class predictions
plt.figure(figsize=(10, 6))
sns.histplot(np.max(calibrated_scores, axis=1), bins=50, kde=True)
plt.axvline(unknown_class_threshold, color='r', linestyle='--', label='Unknown Class Threshold')
plt.title('MetaMax Distribution of Class Probabilities with Unknown Class Threshold')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('MetaMax unknown_class_probability_distribution.png')
plt.show()

# Calculate AUROC for MetaMax's unknown class detection
y_true = (true_labels_full >= num_classes_filtered) 
y_scores = 1 - np.max(calibrated_scores, axis=1) 

auroc = roc_auc_score(y_true, y_scores)
print(f'MetaMax AUROC for unknown class detection: {auroc:.4f}')

'''
# Final archive of GeMOS code 

print("Evaluating GeMOS on the full dataset...")
# Use GeMOS to predict unknown classes
gemos_scores = gemos.predict(features_full_extracted)

# Detect unknown classes
unknown_class_predictions, gemos_scores = gemos.detect_unknown_classes(features_full_extracted)

# Calculate AUROC for unknown class detection
y_true = (true_labels_full >= num_classes_filtered)

# Detect unknown classes with GeMOS
y_scores_gemos = gemos_scores[:, -1]  

# Calculate AUROC for GeMOS's unknown class detection
auroc_gemos = roc_auc_score(y_true, y_scores_gemos)
print(f'GeMOS AUROC for unknown class detection: {auroc_gemos:.4f}')

# Save GeMOS unknown class probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_scores_gemos, bins=50, kde=True)
plt.axvline(unknown_class_threshold, color='r', linestyle='--', label='Unknown Class Threshold')
plt.title('GeMOS Distribution of Class Probabilities with Unknown Class Threshold')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('GeMOS_unknown_class_probability_distribution.png')
plt.show()

# Evaluate OpenMax on the full dataset
print("Evaluating OpenMax on the full dataset...")
openmax_scores = open_max.predict(features_full_extracted)

# Detect unknown classes with OpenMax
unknown_class_predictions_openmax = openmax_scores[:, -1] > unknown_class_threshold
num_unknown_predictions_openmax = np.sum(unknown_class_predictions_openmax)
print(f"Number of unknown class predictions by OpenMax: {num_unknown_predictions_openmax}")

# Calculate AUROC for OpenMax's unknown class detection
y_scores_openmax = openmax_scores[:, -1]  # Probability for the unknown class

# Ensure there are no NaN values in the scores
if np.isnan(y_scores_openmax).any():
    print("Warning: NaN values found in OpenMax scores. Handling NaNs.")
    y_scores_openmax = np.nan_to_num(y_scores_openmax, nan=0.0)

# Example: Ensure consistent lengths between true labels and predicted scores
if len(y_true) != len(y_scores_openmax):
    print(f"Warning: Mismatch in sample sizes between y_true ({len(y_true)}) and y_scores_openmax ({len(y_scores_openmax)})")
    min_length = min(len(y_true), len(y_scores_openmax))
    y_true = y_true[:min_length]
    y_scores_openmax = y_scores_openmax[:min_length]

auroc_openmax = roc_auc_score(y_true, y_scores_openmax)
print(f'OpenMax AUROC for unknown class detection: {auroc_openmax:.4f}')

# Plot OpenMax's unknown class scores
plt.figure(figsize=(10, 6))
sns.histplot(y_scores_openmax, bins=50, kde=True)
plt.axvline(unknown_class_threshold, color='r', linestyle='--', label='Unknown Class Threshold')
plt.title('OpenMax: Distribution of Unknown Class Scores')
plt.xlabel('Unknown Class Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('openmax_unknown_class_score_distribution.png')
plt.show()

'''
