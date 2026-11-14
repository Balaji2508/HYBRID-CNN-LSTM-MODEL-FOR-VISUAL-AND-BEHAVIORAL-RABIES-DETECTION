import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = r"D:\balaji v2\Rabbies_Detection_Project\data\train"

datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
    TRAIN_DIR,
    class_mode='binary',
    batch_size=1
)

print("\n" + "="*30)
print("CLASS MAPPING VERIFICATION")
print("="*30)
print(generator.class_indices)
print("="*30)
