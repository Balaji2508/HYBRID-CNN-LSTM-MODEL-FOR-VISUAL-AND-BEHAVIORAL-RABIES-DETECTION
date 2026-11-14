import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import random

# CONFIG
MODEL_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\rabies_classifier.h5"
TEST_DIR = r"D:\balaji v2\Rabbies_Detection_Project\data\test"
IMG_SIZE = (224, 224)

def predict_image(model, img_path):
    try:
        # Load and preprocess
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Batch dimension
        img_array /= 255.0 # Normalize
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Interpretation
        # Since we used class_mode='binary', typically 0=First Class, 1=Second Class (Alphabetical)
        # Folders: 'normal' vs 'rabid' -> Normal=0, Rabid=1
        
        confidence = prediction
        if prediction > 0.5:
            label = "RABID"
            conf_percent = prediction * 100
        else:
            label = "NORMAL"
            conf_percent = (1 - prediction) * 100
            
        return label, conf_percent
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, 0

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print("\n" + "="*50)
    print("IMAGE MODEL RANDOM TEST")
    print("="*50)
    
    classes = ['normal', 'rabid']
    
    for cls in classes:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
            
        # Get random images
        files = os.listdir(cls_dir)
        if not files:
            continue
            
        params = min(3, len(files))
        selected = random.sample(files, params)
        
        print(f"\nTesting Class: {cls.upper()}")
        print("-" * 30)
        
        for fname in selected:
            path = os.path.join(cls_dir, fname)
            label, conf = predict_image(model, path)
            
            # Check if correct
            is_correct = (label.lower() == cls)
            icon = "✅" if is_correct else "❌"
            
            print(f"Image:      {fname}")
            print(f"Prediction: {label} ({conf:.2f}%) {icon}")

if __name__ == "__main__":
    main()
