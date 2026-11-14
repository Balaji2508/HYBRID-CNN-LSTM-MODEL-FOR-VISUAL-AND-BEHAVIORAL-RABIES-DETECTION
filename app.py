import streamlit as st
import numpy as np
from PIL import Image
from utils import load_models, preprocess_image, preprocess_text, get_safety_precautions
import os

# Config
IMAGE_MODEL_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\rabies_classifier.h5"
TEXT_MODEL_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\text_model.h5"
TOKENIZER_PATH = r"D:\balaji v2\Rabbies_Detection_Project\models\tokenizer.pickle"

# Initialize
st.set_page_config(page_title="Rabies Detection System", page_icon="🐕", layout="wide")

@st.cache_resource
def get_models():
    return load_models(IMAGE_MODEL_PATH, TEXT_MODEL_PATH, TOKENIZER_PATH)

image_model, text_model, tokenizer = get_models()

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Home", "Image Classification", "Text Classification"])

# Home Page
if app_mode == "Home":
    st.title("🐕 AI-Based Rabies Detection System")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Rabid_dog.jpg/640px-Rabid_dog.jpg", use_column_width=True, caption="Rabies Awareness")
    st.markdown("""
    ## Welcome
    This application uses Artificial Intelligence to help detect rabies in dogs.
    
    ### Modules:
    1. **Image Classification**: Upload an image of a dog to detect signs of 'Furious Rabies'.
    2. **Text Classification**: Enter symptoms to detect 'Dumb/Paralytic Rabies'.
    
    > **Disclaimer**: This tool is for educational and assistive purposes only. Always consult a veterinarian or medical professional for definitive diagnosis.
    """)

# Image Classification Page
elif app_mode == "Image Classification":
    st.title("📸 Furious Rabies Detection (Image)")
    st.write("Upload an image of a dog to analyze its behavior/appearance.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner('Analyzing...'):
                if image_model:
                    processed_img = preprocess_image(image)
                    prediction = image_model.predict(processed_img)
                    score = prediction[0][0]
                    # Assuming 1 = Rabid, 0 = Normal? 
                    # Keras ImageDataGenerator 'binary' usually maps alphabetical: Normal=0, Rabid=1.
                    # Verify: 'normal' comes before 'rabid'. So 0=normal, 1=rabid.
                    
                    # Safety Threshold: 0.5 (Balanced)
                    THRESHOLD = 0.5
                    
                    label = "Rabid" if score > THRESHOLD else "Normal"
                    
                    # Display confidence based on the predicted class
                    if label == "Rabid":
                        # If score is 0.4 (and threshold 0.3), we are confident it's Rabid relative to our safety line.
                        # But strictly mathematically, it's just the raw score.
                        display_confidence = score 
                    else:
                        display_confidence = 1 - score
                    
                    st.subheader(f"Prediction: {label}")
                    st.write(f"Confidence: {display_confidence:.2%}")
                    st.info(f"Raw Model Probability (0=Normal, 1=Rabid): {score:.4f}")
                    
                    precautions = get_safety_precautions(label, display_confidence)
                    st.warning("Safety Precautions:")
                    for p in precautions:
                        st.write(f"- {p}")
                else:
                    st.error("Model not loaded yet. Please wait or check logs. Training might be in progress.")

# Text Classification Page
elif app_mode == "Text Classification":
    st.title("📝 Dumb/Paralytic Rabies Detection (Text)")
    st.write("Describe the dog's symptoms or behavior.")
    
    user_input = st.text_area("Enter Description (e.g., 'The dog is parsing huge amounts of saliva and looks confused')", height=150)
    
    if st.button("Analyze Text"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner('Analyzing...'):
                if text_model and tokenizer:
                    processed_text = preprocess_text(user_input, tokenizer)
                    prediction = text_model.predict(processed_text)
                    # Prediction is softmax [prob_class0, prob_class1]
                    # Classes in pd.get_dummies are usually alphabetical.
                    # 'dumb_rabies' vs 'normal'. D comes before N.
                    # So 0=dumb_rabies, 1=normal.
                    
                    pred_class_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    if pred_class_idx == 0:
                         pred_label = "Rabid (Dumb/Paralytic)"
                         simple_label = "Rabid"
                    else:
                         pred_label = "Normal"
                         simple_label = "Normal"
                    
                    st.subheader(f"Prediction: {pred_label}")
                    st.write(f"Confidence: {confidence:.2%}")
                    
                    precautions = get_safety_precautions(simple_label, confidence)
                    st.warning("Safety Precautions:")
                    for p in precautions:
                        st.write(f"- {p}")
                else:
                    st.error("Model not loaded. Please wait or check logs.")
