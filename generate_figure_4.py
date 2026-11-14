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
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Y-center
yc = 3.0

# 1. Input: Token Sequence
draw_box(ax, 0.5, 2.5, 2.0, 1.0, "Input Tokens\n(Length 100)", color='#E3F2FD')

# Arrow
draw_arrow(ax, 2.5, yc, 3.5, yc)

# 2. Embedding Layer
draw_box(ax, 3.5, 2.0, 2.0, 2.0, "Embedding\n(5000, 100)", color='#C8E6C9')

# Arrow
draw_arrow(ax, 5.5, yc, 6.5, yc)

# 3. Spatial Dropout
draw_box(ax, 6.5, 2.25, 1.5, 1.5, "Spatial\nDropout 1D\n(0.2)", color='#FFCC80')

# Arrow
draw_arrow(ax, 8.0, yc, 9.0, yc)

# 4. LSTM Layer
draw_box(ax, 9.0, 2.0, 2.0, 2.0, "LSTM\n(100 units)\nDropout", color='#D1C4E9')

# Arrow
draw_arrow(ax, 11.0, yc, 12.0, yc)

# 5. Output Layer
draw_box(ax, 12.0, 2.5, 1.5, 1.0, "Dense\n(Softmax)", color='#FFAB91')

# Output Labels
ax.text(12.75, 1.5, "Rabid / Normal", ha='center', va='top', fontsize=9)

output_path = "lstm_architecture.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {os.path.abspath(output_path)}")
