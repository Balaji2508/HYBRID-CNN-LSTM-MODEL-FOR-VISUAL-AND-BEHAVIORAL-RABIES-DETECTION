import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import numpy as np
import pickle
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# Function to load models
def load_models(image_model_path, text_model_path, tokenizer_path):
    try:
        # Load Image Model
        if os.path.exists(image_model_path):
            image_model = load_model(image_model_path)
        else:
            image_model = None
            
        # Load Text Model
        if os.path.exists(text_model_path):
            text_model = load_model(text_model_path)
        else:
            text_model = None
        
        # Load Tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            tokenizer = None
            
        return image_model, text_model, tokenizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Preprocessing for Image
def preprocess_image(image, target_size=(224, 224)):
    # Convert PIL image to array
    image = np.array(image)
    # Resize
    image = cv2.resize(image, target_size)
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
    # Scale
    # OLD: image = image / 255.0  <-- INCORRECT for ResNet50
    # NEW: use preprocess_input (handling BGR conversion, mean subtraction)
    
    # Expand dims first (preprocess_input expects batch)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Preprocessing for Text
def preprocess_text(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    return padded

# Safety Precaution Module
def get_safety_precautions(prediction_label, confidence):
    if prediction_label == "Rabid":
        return [
            "**DANGER: STAY AWAY!** Do not approach the dog.",
            "Call animal control or local authorities immediately.",
            "Isolate the dog if possible without risking exposure.",
            "Do not attempt to feed or touch the dog.",
            "Seek medical attention immediately if bitten or scratched."
        ]
    elif prediction_label == "Normal":
         return [
            "The dog appears normal, but always exercise caution with strange animals.",
            "Ensure effective vaccination against rabies.",
            "Regular veterinary check-ups are recommended."
        ]
    else:
        return ["Unable to determine safety precautions."]
