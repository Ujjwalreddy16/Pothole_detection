import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
img_size = 224
batch_size = 32
epochs = 20
learning_rate = 0.0001
train_dir = r"D:\Projects\Python\DS\Dataset\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Train data"   # <-- Replace with your actual path
test_dir = r"D:\Projects\Python\DS\Dataset\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Dataset 1 (Simplex)\Test data"     # <-- For final testing (not now)

# --- DATA AUGMENTATION WITH VALIDATION SPLIT ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.24,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'    # IMPORTANT
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # IMPORTANT
)

# --- MODEL BUILDING ---
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- MODEL TRAINING ---
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# --- UNFREEZE AND FINE-TUNE ---
base_model.trainable = True
optimizer_fine = Adam(learning_rate=learning_rate/10)
model.compile(optimizer=optimizer_fine, loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
history_fine = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator
)

# --- SAVE MODEL ---
model.save('pothole_detection_efficientnetb0.h5')

# --- PLOTTING TRAINING HISTORY ---
def plot_history(histories, titles):
    plt.figure(figsize=(14, 6))
    
    for i, history in enumerate(histories):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title(titles[i])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

plot_history([history, history_fine], ['Initial Training', 'Fine Tuning'])

print("✅ Model trained and saved successfully!")

mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)         # type: ignore
mobilenet_base.trainable = False
mobilenet_features = GlobalAveragePooling2D()(mobilenet_base.output)

# EfficientNetB0 branch
efficientnet_base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)          # type: ignore
efficientnet_base.trainable = False
efficientnet_features = GlobalAveragePooling2D()(efficientnet_base.output)

# Concatenate features
combined_features = concatenate([mobilenet_features, efficientnet_features])
x = Dropout(0.5)(combined_features)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)