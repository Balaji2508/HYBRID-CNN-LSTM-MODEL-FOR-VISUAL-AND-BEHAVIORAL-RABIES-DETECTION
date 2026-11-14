import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
try:
    from tensorflow.keras.applications import ResNet50
except ImportError:
    from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Config
DATA_DIR = r"D:\balaji v2\Rabbies_Detection_Project\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_SAVE_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\rabies_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    if not os.path.exists(TRAIN_DIR):
        print(f"Train directory not found: {TRAIN_DIR}")
        return

    # Data Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # Correct preprocessing for ResNet50
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    # Build Model
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    # Callbacks
    # Ensure models dir exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    # Train
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("\nInitial training complete. Starting Fine-Tuning...")
    
    # FINE-TUNING PHASE
    # Unfreeze loosely
    for layer in model.layers:
        layer.trainable = True
        
    # Freeze bottom N layers
    # ResNet50+Head has many layers. We freeze until the last 30.
    total_layers = len(model.layers)
    freeze_until = total_layers - 30
    
    for layer in model.layers[:freeze_until]:
        layer.trainable = False

    # Need to recompile with very low learning rate
    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Train again (Fine-tuning)
    fine_tune_epochs = EPOCHS + 20
    history_fine = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training complete. Model saved to", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
