import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set Font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def draw_box(ax, x, y, width, height, text, color='#E0E0E0', ec='black'):
    # Use FancyBboxPatch for rounded corners
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor=ec, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, fontweight='bold', wrap=True)
    return rect

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

# Setup Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 1. Inputs (Left)
draw_box(ax, 0.5, 7.5, 2.0, 1.0, "Image Inputs\n(Video Frames)", color='#BBDEFB')
draw_box(ax, 0.5, 3.5, 2.0, 1.0, "Text Input\n(Symptom Desc.)", color='#BBDEFB')

# 2. Models (Center)
draw_box(ax, 3.5, 7.0, 2.5, 2.0, "Visual Classification\nEngine\n(ResNet50 + Custom Head)", color='#C8E6C9')
draw_box(ax, 3.5, 3.0, 2.5, 2.0, "Semantic Classification\nEngine\n(LSTM + Embedding)", color='#C8E6C9')

# 3. Outputs (Right-Center)
draw_box(ax, 7.0, 7.5, 2.0, 1.0, "Risk Class + Score\n(Probabilistic)", color='#FFF9C4')
draw_box(ax, 7.0, 3.5, 2.0, 1.0, "Risk Class\n(Binary)", color='#FFF9C4')

# 4. Streamlit Interface (Far Right / Bottom Wrapper)
# Let's put a big box around outputs or a final consolidation block
draw_box(ax, 4.0, 0.5, 4.0, 1.5, "Streamlit Interface Layer\n(Consolidation & Display)", color='#E1BEE7')

# Arrows - Image Path
draw_arrow(ax, 2.5, 8.0, 3.5, 8.0) # Input -> Model
draw_arrow(ax, 6.0, 8.0, 7.0, 8.0) # Model -> Output
draw_arrow(ax, 8.0, 7.5, 6.0, 2.0) # Output -> Streamlit (Curved?)
# Let's do straight lines for now or manual points
ax.plot([8.0, 8.0, 6.0], [7.5, 6.5, 2.0], color='black', lw=1.5, zorder=0) # Path to interface
# Fix schematic later if needed. Let's direct outputs to Interface directly.

# Redrawing clearer paths
ax.clear()
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Re-defining positions for better flow

# Inputs
input_w, input_h = 2.0, 1.0
model_w, model_h = 2.5, 1.5
out_w, out_h = 2.0, 1.0

# Rows
y_img = 7
y_txt = 4

# Cols
x_in = 1
x_mod = 4.5
x_out = 8

# Draw Blocks
# Image Row
draw_box(ax, x_in, y_img, input_w, input_h, "Image Input\n(User Upload)", color='#E3F2FD')
draw_box(ax, x_mod, y_img-0.25, model_w, model_h, "Visual Engine\n(ResNet50)", color='#E8F5E9')
draw_box(ax, x_out, y_img, out_w, out_h, "Risk Score\n(0-100%)", color='#FFF8E1')

# Text Row
draw_box(ax, x_in, y_txt, input_w, input_h, "Text Input\n(Symptom Desc.)", color='#E3F2FD')
draw_box(ax, x_mod, y_txt-0.25, model_w, model_h, "Semantic Engine\n(LSTM)", color='#E8F5E9')
draw_box(ax, x_out, y_txt, out_w, out_h, "Class Prediction\n(Rabid vs Normal)", color='#FFF8E1')

# Integration Layer (Bottom)
draw_box(ax, 3.5, 0.5, 6.0, 1.5, "Streamlit Interface Layer\n(Consolidation & Protocol Recommendations)", color='#F3E5F5')

# Arrows
# Image Flow
draw_arrow(ax, x_in + input_w, y_img + input_h/2, x_mod, y_img + input_h/2)
draw_arrow(ax, x_mod + model_w, y_img + input_h/2, x_out, y_img + input_h/2)

# Text Flow
draw_arrow(ax, x_in + input_w, y_txt + input_h/2, x_mod, y_txt + input_h/2)
draw_arrow(ax, x_mod + model_w, y_txt + input_h/2, x_out, y_txt + input_h/2)

# Consolidation Flow (Downwards)
# From Image Output to Streamlit
draw_arrow(ax, x_out + out_w/2, y_img, x_out + out_w/2, 2.5) # Vertical down
ax.plot([x_out + out_w/2, 6.5], [2.5, 2.0], color='black', lw=1.5) # Connect to box
ax.annotate("", xy=(6.5, 2.0), xytext=(x_out + out_w/2, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))

# From Text Output to Streamlit
draw_arrow(ax, x_out + out_w/2, y_txt, x_out + out_w/2, 2.5) # Merge with above line actually
# Just draw arrow from Text Output bottom to Streamlit top
draw_arrow(ax, x_out + out_w/2, y_txt, 7.5, 2.0)

# Also Image Output to Streamlit directly
draw_arrow(ax, x_out + out_w/2, y_img, 7.5, 2.0)

# Title is handled by caption in paper, but we can add a simple label if needed. 
# User asked for Times New Roman, handled globally.

output_path = "system_architecture.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {os.path.abspath(output_path)}")
