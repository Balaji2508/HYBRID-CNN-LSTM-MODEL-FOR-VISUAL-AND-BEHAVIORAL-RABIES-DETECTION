import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Set Font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Paths
base_dir = r"d:\balaji v2\Rabbies_Detection_Project\data\train"
rabid_img_name = "img_g_001_mp4-0072_jpg.rf.fa83379d7b2ee925336819c8d28034ee.jpg"
normal_img_name = "img_n02085620_10074.jpg"

rabid_path = os.path.join(base_dir, "rabid", rabid_img_name)
normal_path = os.path.join(base_dir, "normal", normal_img_name)

def process_and_visualize(img_path, class_label, ax_row, simulated_original_size):
    # Load image
    # We load it, but we "simulate" a raw input by resizing it to a random larger layout
    # This represents what the pipeline would receive from a real camera
    real_img = load_img(img_path)
    real_array = img_to_array(real_img) / 255.0
    
    # Simulate "Original" raw input (e.g. video frame size)
    # We use TF resize to upsample validly just for visualization
    simulated_raw = tf.image.resize(real_array, simulated_original_size)
    
    # 1. Display "Original" (Simulated Raw)
    ax_row[0].imshow(simulated_raw)
    ax_row[0].set_title(f"Original Input\n({simulated_original_size[0]}x{simulated_original_size[1]})")
    ax_row[0].axis('off')
    
    # Add Class Label
    if class_label:
        ax_row[0].text(-0.2, 0.5, class_label, transform=ax_row[0].transAxes, 
                       rotation=90, va='center', ha='right', fontsize=14, fontweight='bold')

    # 2. Resized Image (The Pipeline Step)
    target_size = (224, 224)
    resized_img = tf.image.resize(simulated_raw, target_size)
    
    ax_row[1].imshow(resized_img)
    ax_row[1].set_title(f"Resized\n(224x224)")
    ax_row[1].axis('off')

    # 3. Augmented Image
    # Apply visible changes: Horizontal Flip + Rotation + Zoom
    # We take the resized image and apply transformations
    
    # Rotation (15 deg) + Zoom (0.8) + Shift
    augmented_img = tf.keras.preprocessing.image.apply_affine_transform(
        resized_img.numpy(), 
        theta=20, 
        tx=30, 
        ty=10, 
        zx=0.9,
        zy=0.9,
        fill_mode='nearest'
    )
    
    # Explicit Horizontal Flip for visualization if it wasn't flipped by affine
    # Let's just flip it to be sure it looks different
    augmented_img = tf.image.flip_left_right(augmented_img)
    
    ax_row[2].imshow(augmented_img)
    ax_row[2].set_title(f"Augmented\n(Flip + Rot + Shift)")
    ax_row[2].axis('off')

# Setup Plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Process Rabid (Simulate distinct "video frame" aspect ratio)
process_and_visualize(rabid_path, "Furious Rabies", axes[0], simulated_original_size=(300, 450))

# Process Normal (Simulate standard "photo" aspect ratio)
process_and_visualize(normal_path, "Normal Case", axes[1], simulated_original_size=(400, 500))

plt.tight_layout()
plt.subplots_adjust(left=0.1, wspace=0.3)

output_path = "preprocessing_pipeline.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {os.path.abspath(output_path)}")
