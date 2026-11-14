import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def generate_confusion_matrix():
    # Paths
    val_dir = r"d:\balaji v2\Rabbies_Detection_Project\data\val"
    model_path = r"d:\balaji v2\Rabbies_Detection_Project\image_model\rabies_resnet50_finetuned.h5"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load Model
    print("Loading model...")
    model = load_model(model_path)

    # Data Generator
    print("Preparing data generator...")
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False  # Important for confusion matrix
    )

    # Predict
    print("Running inference...")
    y_pred_prob = model.predict(val_generator, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = val_generator.classes

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("Confusion Matrix:")
    print(cm)
    print("Normalized:")
    print(cm_normalized)

    # Plot
    plt.figure(figsize=(8, 6))
    
    # Class labels
    class_names = ['Normal', 'Rabid']

    # Create heatmap
    # annot=True shows the values
    # fmt='.2f' for float 2 decimal places
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "fontfamily": "Times New Roman"})

    plt.title('Normalized Confusion Matrix (ResNet50)', fontsize=14, fontname='Times New Roman', pad=15)
    plt.ylabel('True Label', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Predicted Label', fontsize=12, fontname='Times New Roman')
    
    # Customizing tick labels
    plt.xticks(fontsize=11, fontname='Times New Roman')
    plt.yticks(fontsize=11, fontname='Times New Roman', rotation=0)

    # Save
    output_path = "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_confusion_matrix()
