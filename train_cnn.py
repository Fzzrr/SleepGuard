import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Path ke dataset — pastikan folder Eye_Dataset/open dan Eye_Dataset/closed
dataset_path = 'Eye_Dataset'

# 2. Image Augmentation dan Normalisasi
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=False,
    validation_split=0.2
)

# 3. Data Generator
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(24, 24),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(24, 24),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    subset='validation'
)

# 4. Arsitektur CNN yang Lebih Dalam & Stabil
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 5. Kompilasi Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Callback: EarlyStopping dan Simpan Model Terbaik
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint('best_eye_model.h5', save_best_only=True, monitor='val_loss')
]

# 7. Training
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks
)

# 8. Simpan Model Final
model.save('eye_status_model_final.h5')
print("✅ Model final disimpan sebagai 'eye_status_model_final.h5'")
