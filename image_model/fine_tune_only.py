import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Config
DATA_DIR = r"D:\balaji v2\Rabbies_Detection_Project\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_SAVE_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\rabies_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 20

def main():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model not found at {MODEL_SAVE_PATH}")
        return

    print(f"Loading model from {MODEL_SAVE_PATH}...")
    model = load_model(MODEL_SAVE_PATH)
    
    # FINE-TUNING SETUP
    print("Setting up for fine-tuning...")
    
    # Unfreeze the base model (ResNet50)
    # Since we don't know exactly where ResNet layers are in the flat logical model,
    # we iterate and unfreeze broadly, but we want to freeze the bottom N layers.
    # ResNet50 has ~175 layers. We want to unfreeze the top ~30.
    
    total_layers = len(model.layers)
    print(f"Total layers in model: {total_layers}")
    
    # Unfreeze all first
    for layer in model.layers:
        layer.trainable = True
        
    # Freeze the bottom (Total - 30) layers
    freeze_until = total_layers - 30
    for layer in model.layers[:freeze_until]:
        layer.trainable = False
        
    print(f"Frozen layers: 0 to {freeze_until}")
    print(f"Trainable layers: {freeze_until} to {total_layers}")

    # Recompile with low learning rate
    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Data Generators (Same as before)
    # Data Generators (Same as before)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
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
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    
    print("Starting Fine-Tuning Training...")
    history = model.fit(
        train_generator,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Fine-tuning complete. Model updated.")

if __name__ == "__main__":
    main()
