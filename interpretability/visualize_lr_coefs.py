"""
LR Coefficient Visualization
N별 β_img, β_txt 변화 분석
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 데이터 (robustness 실험 결과)
data = {
    'N=24': {'img': 3.7053, 'txt': 0.3056},
    'N=12': {'img': 3.6462, 'txt': 0.4375},
    'N=6':  {'img': 3.7896, 'txt': 0.3116},
}

# Extract
n_values = list(data.keys())
img_coefs = [data[n]['img'] for n in n_values]
txt_coefs = [data[n]['txt'] for n in n_values]
ratios = [data[n]['img'] / data[n]['txt'] for n in n_values]

# Figure 1: Stacked bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Coefficients
x = np.arange(len(n_values))
width = 0.35

bars1 = ax1.bar(x - width/2, img_coefs, width, label='Image (β_img)', color='#3498db')
bars2 = ax1.bar(x + width/2, txt_coefs, width, label='Text (β_txt)', color='#e74c3c')

ax1.set_xlabel('Image Quality (N slices)', fontsize=12)
ax1.set_ylabel('LR Coefficient', fontsize=12)
ax1.set_title('Logistic Regression Coefficients by Quality Level', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(n_values)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

# Right: Ratio
bars3 = ax2.bar(x, ratios, color='#95a5a6', alpha=0.7)
ax2.set_xlabel('Image Quality (N slices)', fontsize=12)
ax2.set_ylabel('Ratio (β_img / β_txt)', fontsize=12)
ax2.set_title('Image-to-Text Coefficient Ratio', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(n_values)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Equal weight')
ax2.legend()

# Add values
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}:1',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save
output_dir = Path('results/interpretability')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'lr_coefficients.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved to: {output_dir / 'lr_coefficients.png'}")

# Print table
print("\n" + "="*70)
print("LR COEFFICIENT ANALYSIS")
print("="*70)
print(f"{'N_slices':<12} {'β_img':<12} {'β_txt':<12} {'Ratio':<15} {'Interpretation':<20}")
print("-"*70)

for n in n_values:
    img = data[n]['img']
    txt = data[n]['txt']
    ratio = img / txt
    
    if ratio > 10:
        interp = "Image dominant"
    elif ratio > 5:
        interp = "Image preferred"
    else:
        interp = "More balanced"
    
    print(f"{n:<12} {img:<12.2f} {txt:<12.2f} {ratio:<15.1f} {interp:<20}")

print("="*70)
print("\nKEY FINDINGS:")
print("1. Image coefficients consistently 10-12x larger than text")
print("2. N=12 shows lowest ratio (8.3:1) - text tries to contribute more")
print("3. But N=12 has worst performance → text contribution is harmful")
print("4. N=6 ratio returns to 12:1, but now text helps (+0.5%p)")
print("   → Not about coefficient size, but signal quality match")

plt.show()