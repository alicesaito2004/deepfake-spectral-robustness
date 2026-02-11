"""
Generate detailed architecture flowcharts for the three classifiers.
Includes all layer specifications, dimensions, and parameter counts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

OUTPUT_DIR = "outputs/architecture"


def draw_box(ax, x, y, width, height, text, color, fontsize=9, edgewidth=1.5):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='black', linewidth=edgewidth)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))


def create_detailed_backbone():
    """Create detailed backbone with all layer specifications."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('SimpleCNNBackbone — Detailed Architecture\n(4 Blocks, ~590K parameters per branch)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Colors
    conv_color = '#74b9ff'
    bn_color = '#81ecec'
    relu_color = '#55efc4'
    pool_color = '#ffeaa7'
    
    # Block specifications
    blocks = [
        {"in": "C", "out": 32, "size_in": 256, "size_out": 128, "params": "C×3×3×32 + 32"},
        {"in": 32, "out": 64, "size_in": 128, "size_out": 64, "params": "32×3×3×64 + 64"},
        {"in": 64, "out": 128, "size_in": 64, "size_out": 32, "params": "64×3×3×128 + 128"},
        {"in": 128, "out": 256, "size_in": 32, "size_out": 16, "params": "128×3×3×256 + 256"},
    ]
    
    y_base = 4
    
    for i, block in enumerate(blocks):
        x_start = 1 + i * 3.8
        
        # Block header
        ax.text(x_start + 1.5, 7.2, f'Block {i+1}', ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Input dimension
        if i == 0:
            ax.text(x_start - 0.3, y_base + 1.5, f'Input:\nC×256×256', ha='center', fontsize=8, style='italic')
        
        # Conv2d
        conv_text = f'Conv2d\n{block["in"]}→{block["out"]}\n3×3, pad=1'
        draw_box(ax, x_start + 0.5, y_base + 1.5, 1.3, 1.4, conv_text, conv_color, 8)
        
        # BatchNorm
        bn_text = f'BatchNorm2d\n({block["out"]})'
        draw_box(ax, x_start + 2, y_base + 1.5, 1.2, 1, bn_text, bn_color, 8)
        
        # ReLU
        draw_box(ax, x_start + 2, y_base, 1, 0.7, 'ReLU', relu_color, 8)
        
        # MaxPool
        pool_text = f'MaxPool2d\n2×2, stride=2'
        draw_box(ax, x_start + 0.5, y_base - 1.2, 1.3, 0.9, pool_text, pool_color, 8)
        
        # Output dimension
        dim_text = f'{block["out"]}×{block["size_out"]}×{block["size_out"]}'
        ax.text(x_start + 2.2, y_base - 1.2, dim_text, ha='center', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
        
        # Arrows within block
        draw_arrow(ax, (x_start + 1.15, y_base + 1.5), (x_start + 1.4, y_base + 1.5))
        draw_arrow(ax, (x_start + 2, y_base + 1), (x_start + 2, y_base + 0.35))
        draw_arrow(ax, (x_start + 1.5, y_base), (x_start + 1.15, y_base - 0.75))
        
        # Arrow to next block
        if i < 3:
            draw_arrow(ax, (x_start + 2.8, y_base - 1.2), (x_start + 3.5, y_base - 1.2))
    
    # Final output box
    draw_box(ax, 15, y_base - 1.2, 1.3, 1, 'Output\n256×16×16', '#dfe6e9', 9)
    draw_arrow(ax, (14.2, y_base - 1.2), (14.35, y_base - 1.2))
    
    # Legend / Notes
    ax.text(8, 1, 'C = input channels (3 for RGB pixel, 1 for grayscale spectrum)', 
            ha='center', fontsize=10, style='italic')
    ax.text(8, 0.4, 'Total: 4 Conv blocks, each with Conv2d → BatchNorm → ReLU → MaxPool', 
            ha='center', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "backbone_detailed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_pixel_classifier_detailed():
    """Create detailed pixel classifier flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Pixel Classifier — Full Architecture\nInput: RGB Image | Output: Real/Fake', 
                 fontsize=14, fontweight='bold', pad=20)
    
    y = 2.8
    
    # Input
    draw_box(ax, 1.2, y, 1.6, 1.4, 'Input\nRGB Image\n3×256×256', '#ff7675', 9)
    
    # Backbone blocks
    blocks = [
        ('Block 1\nConv 3→32\nBN, ReLU\nPool 2×2', '32×128×128'),
        ('Block 2\nConv 32→64\nBN, ReLU\nPool 2×2', '64×64×64'),
        ('Block 3\nConv 64→128\nBN, ReLU\nPool 2×2', '128×32×32'),
        ('Block 4\nConv 128→256\nBN, ReLU\nPool 2×2', '256×16×16'),
    ]
    
    x_pos = 3.2
    for i, (block_text, dim) in enumerate(blocks):
        draw_box(ax, x_pos, y, 1.6, 1.6, block_text, '#74b9ff', 8)
        ax.text(x_pos, y - 1.2, dim, ha='center', fontsize=7, 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
        if i < 3:
            draw_arrow(ax, (x_pos + 0.8, y), (x_pos + 1.0, y))
        x_pos += 1.8
    
    # Global Average Pool
    draw_box(ax, 11, y, 1.2, 1.2, 'GAP\n256×1×1', '#ffeaa7', 9)
    
    # Flatten
    draw_box(ax, 12.5, y, 1, 1.2, 'Flatten\n256', '#81ecec', 9)
    
    # Linear
    draw_box(ax, 14, y, 1.2, 1.2, 'Linear\n256→2', '#55efc4', 9)
    
    # Output
    draw_box(ax, 15.3, y, 1, 1.2, 'Output\nReal/\nFake', '#dfe6e9', 8)
    
    # Arrows
    draw_arrow(ax, (2, y), (2.4, y))
    draw_arrow(ax, (10.2, y), (10.4, y))
    draw_arrow(ax, (11.6, y), (12, y))
    draw_arrow(ax, (13, y), (13.4, y))
    draw_arrow(ax, (14.6, y), (14.8, y))
    
    # Parameter count
    ax.text(8, 0.6, 'Total Parameters: ~591K (backbone) + 514 (classifier) = ~592K', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "pixel_classifier_detailed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_spectrum_classifier_detailed():
    """Create detailed spectrum classifier flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Spectrum Classifier — Full Architecture\nInput: RGB Image → FFT → Magnitude Spectrum | Output: Real/Fake', 
                 fontsize=14, fontweight='bold', pad=20)
    
    y = 2.8
    
    # Input
    draw_box(ax, 1, y, 1.4, 1.4, 'Input\nRGB Image\n3×256×256', '#ff7675', 8)
    
    # FFT Processing
    draw_box(ax, 2.6, y, 1.2, 1.4, 'Grayscale\n1×256×256', '#dfe6e9', 8)
    draw_box(ax, 4.1, y, 1.2, 1.4, 'FFT2D\nfftshift', '#a29bfe', 8)
    draw_box(ax, 5.6, y, 1.2, 1.4, 'Log Mag\n1×256×256', '#fd79a8', 8)
    
    # Backbone blocks  
    blocks = [
        ('Block 1\nConv 1→32\nBN, ReLU\nPool 2×2', '32×128×128'),
        ('Block 2\nConv 32→64\nBN, ReLU\nPool 2×2', '64×64×64'),
        ('Block 3\nConv 64→128\nBN, ReLU\nPool 2×2', '128×32×32'),
        ('Block 4\nConv 128→256\nBN, ReLU\nPool 2×2', '256×16×16'),
    ]
    
    x_pos = 7.2
    for i, (block_text, dim) in enumerate(blocks):
        draw_box(ax, x_pos, y, 1.4, 1.6, block_text, '#74b9ff', 7)
        ax.text(x_pos, y - 1.2, dim, ha='center', fontsize=6, 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
        if i < 3:
            draw_arrow(ax, (x_pos + 0.7, y), (x_pos + 0.9, y))
        x_pos += 1.6
    
    # Final layers
    draw_box(ax, 13.8, y, 1, 1.2, 'GAP\n256', '#ffeaa7', 8)
    draw_box(ax, 15, y, 1.2, 1.2, 'Linear\n256→2', '#55efc4', 8)
    
    # Arrows
    draw_arrow(ax, (1.7, y), (2, y))
    draw_arrow(ax, (3.2, y), (3.5, y))
    draw_arrow(ax, (4.7, y), (5, y))
    draw_arrow(ax, (6.2, y), (6.5, y))
    draw_arrow(ax, (13, y), (13.3, y))
    draw_arrow(ax, (14.3, y), (14.4, y))
    
    # Notes
    ax.text(8, 0.6, 'Total Parameters: ~590K (1-channel backbone) + 514 = ~590K', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "spectrum_classifier_detailed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_dual_classifier_detailed():
    """Create detailed dual-branch classifier flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Dual-Branch Classifier — Full Architecture\nCombines Pixel + Spectrum Features | Output: Real/Fake', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Input
    draw_box(ax, 1.5, 4, 2, 1.5, 'Input\nRGB Image\n3×256×256', '#ff7675', 10)
    
    # ===== PIXEL BRANCH (top) =====
    y_top = 6
    ax.text(6, 7.3, 'Pixel Branch (3-channel input)', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#74b9ff', alpha=0.3))
    
    draw_box(ax, 4, y_top, 1.3, 1, 'Block 1\n3→32', '#74b9ff', 8)
    draw_box(ax, 5.5, y_top, 1.3, 1, 'Block 2\n32→64', '#74b9ff', 8)
    draw_box(ax, 7, y_top, 1.3, 1, 'Block 3\n64→128', '#74b9ff', 8)
    draw_box(ax, 8.5, y_top, 1.3, 1, 'Block 4\n128→256', '#74b9ff', 8)
    draw_box(ax, 10, y_top, 1, 1, 'GAP\n256', '#ffeaa7', 8)
    
    # Pixel branch arrows
    draw_arrow(ax, (2.5, 4.5), (3.35, y_top))
    draw_arrow(ax, (4.65, y_top), (4.85, y_top))
    draw_arrow(ax, (6.15, y_top), (6.35, y_top))
    draw_arrow(ax, (7.65, y_top), (7.85, y_top))
    draw_arrow(ax, (9.15, y_top), (9.5, y_top))
    
    # ===== SPECTRUM BRANCH (bottom) =====
    y_bot = 2
    ax.text(6, 0.7, 'Spectrum Branch (1-channel input)', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#a29bfe', alpha=0.3))
    
    draw_box(ax, 3.2, y_bot, 1, 0.9, 'FFT\nMag', '#a29bfe', 8)
    draw_box(ax, 4.5, y_bot, 1.3, 1, 'Block 1\n1→32', '#74b9ff', 8)
    draw_box(ax, 6, y_bot, 1.3, 1, 'Block 2\n32→64', '#74b9ff', 8)
    draw_box(ax, 7.5, y_bot, 1.3, 1, 'Block 3\n64→128', '#74b9ff', 8)
    draw_box(ax, 9, y_bot, 1.3, 1, 'Block 4\n128→256', '#74b9ff', 8)
    draw_box(ax, 10.5, y_bot, 1, 1, 'GAP\n256', '#ffeaa7', 8)
    
    # Spectrum branch arrows
    draw_arrow(ax, (2.5, 3.5), (2.7, y_bot))
    draw_arrow(ax, (3.7, y_bot), (3.85, y_bot))
    draw_arrow(ax, (5.15, y_bot), (5.35, y_bot))
    draw_arrow(ax, (6.65, y_bot), (6.85, y_bot))
    draw_arrow(ax, (8.15, y_bot), (8.35, y_bot))
    draw_arrow(ax, (9.65, y_bot), (10, y_bot))
    
    # ===== FUSION =====
    draw_box(ax, 12, 4, 1.5, 1.5, 'Concat\n256+256\n= 512', '#81ecec', 9)
    draw_box(ax, 14, 4, 1.5, 1.5, 'Linear\n512 → 2', '#55efc4', 10)
    draw_box(ax, 15.5, 4, 1, 1, 'Output\nReal/\nFake', '#dfe6e9', 8)
    
    # Fusion arrows
    draw_arrow(ax, (10.5, y_top), (11.25, 4.5))
    draw_arrow(ax, (11, y_bot), (11.25, 3.5))
    draw_arrow(ax, (12.75, 4), (13.25, 4))
    draw_arrow(ax, (14.75, 4), (15, 4))
    
    # Parameter count
    ax.text(8, -0.3, 'Total Parameters: ~591K (pixel) + ~590K (spectrum) + 1,026 (fusion) = ~1.18M', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "dual_classifier_detailed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_table():
    """Create architecture comparison table."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    ax.set_title('Architecture Comparison Summary', fontsize=16, fontweight='bold', pad=20)
    
    # Table data
    headers = ['Component', 'Pixel Classifier', 'Spectrum Classifier', 'Dual-Branch Classifier']
    data = [
        ['Input', '3×256×256 RGB', '1×256×256 Spectrum', 'Both'],
        ['Preprocessing', 'None', 'Grayscale → FFT → Log Mag', 'Both paths'],
        ['Backbone', '4-block CNN\n32→64→128→256', '4-block CNN\n32→64→128→256', '2× 4-block CNN'],
        ['Feature Dim', '256', '256', '512 (concat)'],
        ['Classifier', 'Linear 256→2', 'Linear 256→2', 'Linear 512→2'],
        ['Parameters', '~592K', '~590K', '~1.18M'],
        ['Output', '16×16 feature map', '16×16 feature map', '2× 16×16 → concat'],
    ]
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.25, 0.25, 0.3]
    )
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4a69bd')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f5f6fa')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "architecture_summary_table.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating detailed architecture diagrams...")
    create_detailed_backbone()
    create_pixel_classifier_detailed()
    create_spectrum_classifier_detailed()
    create_dual_classifier_detailed()
    create_summary_table()
    
    print(f"\nAll diagrams saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
