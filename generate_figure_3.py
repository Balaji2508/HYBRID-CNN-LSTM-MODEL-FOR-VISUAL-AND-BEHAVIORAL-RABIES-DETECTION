import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set Font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

def draw_box(ax, x, y, width, height, text, color='#E0E0E0', ec='black'):
    # Use FancyBboxPatch for rounded corners
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", linewidth=1.5, edgecolor=ec, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)
    return rect

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

# Setup Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Y-center
yc = 4.0

# 1. ResNet50 Base (Left)
# Represented as a large block or stack
# "Pre-trained convolutional base (5 stages, 48 conv layers)"
draw_box(ax, 0.5, 2.0, 3.5, 4.0, "ResNet50 Base\n(Frozen Layers)\n\n~145 Layers\n(Stages 1-4)", color='#BBDEFB')

# 2. Unfrozen Top Layers
draw_box(ax, 4.5, 2.0, 2.0, 4.0, "Top-30 Layers\n(Unfrozen)\n\nStage 5\nFine-tuning", color='#90CAF9', ec='blue')

# Dashed Boundary
ax.plot([4.25, 4.25], [1.5, 6.5], color='red', linestyle='--', linewidth=2)
ax.text(4.25, 7.0, "Fine-Tuning\nBoundary", ha='center', va='center', color='red', fontsize=10)

# Arrow to GAP
draw_arrow(ax, 6.5, yc, 7.5, yc)
ax.text(7.0, yc+0.2, "7x7x2048", ha='center', va='bottom', fontsize=9)

# 3. Global Average Pooling
draw_box(ax, 7.5, 3.0, 1.5, 2.0, "Global Avg\nPooling", color='#C8E6C9')

# Arrow to Dense
draw_arrow(ax, 9.0, yc, 10.0, yc)
ax.text(9.5, yc+0.2, "2048-d", ha='center', va='bottom', fontsize=9)

# 4. Dense Layer
draw_box(ax, 10.0, 3.0, 1.5, 2.0, "Dense\n(512, ReLU)", color='#FFF9C4')

# Arrow to Dropout
draw_arrow(ax, 11.5, yc, 12.0, yc)

# 5. Dropout
draw_box(ax, 12.0, 3.25, 1.5, 1.5, "Dropout\n(0.5)", color='#FFCC80')

# Arrow to Output
draw_arrow(ax, 13.5, yc, 14.0, yc)

# 6. Output Node
draw_box(ax, 14.0, 3.5, 1.5, 1.0, "Output\n(Sigmoid)", color='#FFAB91')

output_path = "resnet50_architecture.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {os.path.abspath(output_path)}")
