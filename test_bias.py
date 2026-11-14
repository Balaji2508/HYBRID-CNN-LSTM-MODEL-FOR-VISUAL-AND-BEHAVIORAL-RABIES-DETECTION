import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import random

# CONFIG
MODEL_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\rabies_classifier.h5"
TEST_DIR = r"D:\balaji v2\Rabbies_Detection_Project\data\test\normal"
IMG_SIZE = (224, 224)

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    files = os.listdir(TEST_DIR)
    # create full paths
    files = [os.path.join(TEST_DIR, f) for f in files]
    
    # Analyze 20 random images
    sample_size = min(20, len(files))
    selected = random.sample(files, sample_size)
    
    print(f"\nAnalyzing {sample_size} NORMAL images...")
    print("-" * 50)
    
    rabid_count = 0
    normal_count = 0
    
    for path in selected:
        try:
            img = load_img(path, target_size=IMG_SIZE)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            score = model.predict(x, verbose=0)[0][0]
            
            # Using 0.5 threshold for standard metric (ignoring the 0.3 app setting for now)
            if score > 0.5:
                pred = "RABID"
                rabid_count += 1
            else:
                pred = "NORMAL"
                normal_count += 1
                
            fname = os.path.basename(path)
            # Shorten fname
            fname = (fname[:15] + '..') if len(fname) > 15 else fname
            print(f"{fname:<20} | Score: {score:.4f} | Pred: {pred}")
            
        except Exception as e:
            print(f"Error: {e}")

    print("-" * 50)
    print(f"Results on NORMAL Set:")
    print(f"Predicted NORMAL: {normal_count}")
    print(f"Predicted RABID:  {rabid_count} (False Positives)")
    
    accuracy = normal_count / sample_size * 100
    print(f"Accuracy on Normal Class: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
