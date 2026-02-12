import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Generate synthetic malware binary data
def generate_malware_sample(malware_type, num_samples=50):
    """
    Generate synthetic binary data representing malware samples.
    Each malware type has characteristic patterns.
    """
    samples = []
    labels = []

    for _ in range(num_samples):
        # Create 64x64 image from binary data
        base_pattern = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        # Add type-specific patterns
        if malware_type == 0:  # Trojan - strong diagonal patterns
            for i in range(0, 64, 8):
                base_pattern[i:i + 2, i:i + 2] = 200
        elif malware_type == 1:  # Ransomware - grid patterns
            base_pattern[::8, :] = 220
            base_pattern[:, ::8] = 220
        elif malware_type == 2:  # Worm - circular patterns
            center = 32
            for i in range(64):
                for j in range(64):
                    if 15 < np.sqrt((i - center) ** 2 + (j - center) ** 2) < 20:
                        base_pattern[i, j] = 180
        else:  # Spyware - random high-intensity clusters
            cluster_centers = np.random.randint(10, 54, (5, 2))
            for cx, cy in cluster_centers:
                base_pattern[cx - 3:cx + 3, cy - 3:cy + 3] = 230

        # Add noise
        noise = np.random.normal(0, 10, (64, 64))
        sample = np.clip(base_pattern + noise, 0, 255).astype(np.uint8)

        samples.append(sample)
        labels.append(malware_type)

    return np.array(samples), np.array(labels)


# Generate dataset
print("Generating synthetic malware dataset...")
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

# Class names
class_names = ['Trojan', 'Ransomware', 'Worm', 'Spyware']

# Generate training data (200 samples)
for malware_type in range(4):
    X, y = generate_malware_sample(malware_type, num_samples=50)
    X_train_list.append(X)
    y_train_list.append(y)

# Generate test data (80 samples)
for malware_type in range(4):
    X, y = generate_malware_sample(malware_type, num_samples=20)
    X_test_list.append(X)
    y_test_list.append(y)

# Combine all samples
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# Shuffle the data
train_indices = np.random.permutation(len(X_train))
test_indices = np.random.permutation(len(X_test))
X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]

# Preprocessing: Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape for CNN input (add channel dimension)
X_train = X_train.reshape(-1, 64, 64, 1)
X_test = X_test.reshape(-1, 64, 64, 1)

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, 4)
y_test_cat = keras.utils.to_categorical(y_test, 4)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training labels shape: {y_train_cat.shape}")

# Build CNN model
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  input_shape=(64, 64, 1), padding='same', name='conv1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),

    # Second convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                  padding='same', name='conv2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),

    # Third convolutional block
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu',
                  padding='same', name='conv3'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),

    # Flatten and fully connected layers
    layers.Flatten(name='flatten'),
    layers.Dense(128, activation='relu', name='fc1'),
    layers.Dropout(0.5, name='dropout'),
    layers.Dense(4, activation='softmax', name='output')
])

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("\nTraining CNN for malware classification...")
history = model.fit(
    X_train, y_train_cat,
    batch_size=32,
    epochs=30,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Display classification report
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('malware_cnn_training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining plots saved as 'malware_cnn_training_history.png'")
plt.show()

# Visualize sample predictions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i in range(8):
    axes[i].imshow(X_test[i].reshape(64, 64), cmap='gray')
    true_label = class_names[y_test[i]]
    pred_label = class_names[y_pred_classes[i]]
    confidence = y_pred[i][y_pred_classes[i]] * 100

    color = 'green' if y_test[i] == y_pred_classes[i] else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                      color=color, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('malware_sample_predictions.png', dpi=300, bbox_inches='tight')
print("Sample predictions saved as 'malware_sample_predictions.png'")
plt.show()

print("\n" + "=" * 60)
print("CNN Malware Classification Complete")
print("=" * 60)