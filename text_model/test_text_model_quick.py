import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

MODEL_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\text_model.h5"
TOKENIZER_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\tokenizer.pickle"

# Load artifacts
print(f"Loading model from {MODEL_PATH}...")
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Labels (Alphabetical order: 'dumb_rabies' comes before 'normal')
LABELS = ['Dumb Rabies', 'Normal']

def predict(text):
    # Preprocess
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    
    # Predict
    pred = model.predict(padded, verbose=0)[0]
    
    # Get class
    class_idx = np.argmax(pred)
    label = LABELS[class_idx]
    confidence = pred[class_idx]
    
    return label, confidence, pred

# Test Cases
test_sentences = [
    "My dog is playing fetch and eating well.",
    "The dog is hiding in dark corners and has a dropped jaw.",
    "He is drooling excessively and can't swallow water.",
    "Dog is barking at strangers but eating normal food.", # Should be Normal/Low Risk for dumb rabies
    "The dog looks tired and is vomiting foam.",
    "My puppy is sleeping a lot but wakes up to play.",
    "Complete paralysis of the hind legs and jaw hanging open.",
    "He is choking on something and making weird sounds." # Classic dumb rabies symptom
]

print("\n" + "="*50)
print("TEXT MODEL PERFORMANCE TEST")
print("="*50)

for text in test_sentences:
    label, conf, raw_pred = predict(text)
    print(f"Input:      '{text}'")
    print(f"Prediction: {label.upper()}")
    print(f"Confidence: {conf:.2%}")
    print(f"Raw Score:  Normal: {raw_pred[0]:.4f}, Rabid: {raw_pred[1]:.4f}")
    if label == "Dumb Rabies":
        print("⚠️  Risk Detected")
    else:
        print("✅  Safe")
    print("-" * 50)
