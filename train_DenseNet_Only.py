import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from DenseNet import DenseNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Function to print ASCII banner
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
Batch_size = 4  # Further reduced batch size to fit GPU memory

# Load IoT-23 dataset with progress bar
file_path = 'iot23_combined_new.csv'
print("Loading dataset...")
df = pd.read_csv(file_path)

# Drop the first column
df = df.drop(df.columns[0], axis=1)
print("First column dropped.")
print("Dataset loaded.")

# Display the first few rows to understand the data structure
print(df.head())
print(df.info())

# Identify features and target column
# Assuming 'label' is the target column and other columns are features
features = df.drop(columns=['label'])
labels = df['label']

# Convert all object columns to string type to avoid mixed types
categorical_columns = features.select_dtypes(include=['object']).columns
for column in tqdm(categorical_columns, desc="Converting categorical columns to strings"):
    features[column] = features[column].astype(str)

# Encode categorical features with Label Encoding
label_encoder = LabelEncoder()
for column in tqdm(categorical_columns, desc="Label encoding categorical columns"):
    features[column] = label_encoder.fit_transform(features[column])

# Encode labels
print("Encoding labels...")
labels_encoded = label_encoder.fit_transform(labels)
print("Labels encoded.")

# Standardize the feature columns
print("Standardizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("Features standardized.")

# Split into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
print("Data split into train and test sets.")

# Convert labels to categorical
print("Converting labels to categorical...")
num_classes = len(np.unique(labels_encoded))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
print("Labels converted to categorical.")

# Calculate new dimensions for reshaping
print("Calculating new dimensions for reshaping")
X_train_reshaped = X_train.reshape(-1, 20, 1, 1)
print(f"Reshaped X_train shape: {X_train_reshaped.shape}")
print(f"Reshaped X_test shape: {X_test_reshaped.shape}")
print("Done")

# Define the model save path
model_save_path = 'densenet_model.keras'

# Check if the model already exists
if os.path.exists(model_save_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_save_path)
    print("Model loaded.")
else:
    # Initialize DenseNet with regularization
    print("Initializing DenseNet")
    input_shape = (20, 1, 1)  # Adjust input shape as necessary
    densenet = DenseNet(input_shape=input_shape, num_classes=num_classes, weight_decay=1e-4)
    model = densenet.get_model()
    model.summary()
    print("Done")

    # Compile the model with a smaller learning rate
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    # Train DenseNet with further reduced batch size and progress bar
    print("Training model...")
    history = model.fit(X_train_reshaped, y_train_cat, epochs=10, batch_size=Batch_size, verbose=1, callbacks=[early_stopping])
    print("Model trained.")

    # Save the model in HDF5 format
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Custom function to load npy files with memory mapping
def load_npy_with_mmap(file_path, desc="Loading"):
    print(f"{desc}...")
    return np.load(file_path, mmap_mode='r')



# Evaluate DenseNet model before MetaMax
print("Evaluating DenseNet model...")
logits = model.predict(X_test_reshaped, batch_size=Batch_size, verbose=1)
predicted = np.argmax(logits, axis=1)
true_labels = np.argmax(y_test_cat, axis=1)

# Calculate accuracy, F1 score, and confusion matrix
accuracy = accuracy_score(true_labels, predicted)
f1 = f1_score(true_labels, predicted, average='weighted')
conf_matrix = confusion_matrix(true_labels, predicted)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for DenseNet Model')
plt.savefig('figure.png')
