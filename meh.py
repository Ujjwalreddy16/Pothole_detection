# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import load_img, img_to_array
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.trainers.data_adapters.py_dataset_adapter")

# Paths
train_data_dir = "D:\Projects\Python\DS\Dataset\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Train data"
test_data_dir = "D:\Projects\Python\DS\Dataset\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Test data"

# ImageDataGenerators
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.24  # 76% train, 24% validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Train generator
train_generator = train_val_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset='training'  # 76%
)

# Validation generator
val_generator = train_val_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset='validation'  # 24%
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Save model
model.save('/kaggle/working/final_model.h5')

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.tight_layout()
plt.show()

# Testing model on Test Data
test_images = []
test_labels = []

test_filenames = os.listdir(test_data_dir)

for fname in test_filenames:
    if fname.lower().endswith(".jpg"):
        img_path = os.path.join(test_data_dir, fname)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)

        if "Positive" in fname:
            test_labels.append(1)
        else:
            test_labels.append(0)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Evaluate model
if len(test_images) > 0:
    model = load_model('/kaggle/working/final_model.h5')
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    predictions = model.predict(test_images)
    predicted_classes = (predictions > 0.5).astype(int)

    mse = np.mean((test_labels - predictions.squeeze())**2)
    rmse = np.sqrt(mse)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
else:
    print("Test set is empty or could not be loaded properly.")
