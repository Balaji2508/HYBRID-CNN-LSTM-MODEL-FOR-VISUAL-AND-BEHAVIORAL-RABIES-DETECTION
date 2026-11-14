import os
import sys

# Suppress TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TRAIN_DIR = r"D:\balaji v2\Rabbies_Detection_Project\data\train"

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: {TRAIN_DIR} not found")
    sys.exit(1)

datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(10, 10), # Speed
    class_mode='binary',
    batch_size=1,
    shuffle=False
)

print("\n!!! MAPPING START !!!")
print(generator.class_indices)
print("!!! MAPPING END !!!")
