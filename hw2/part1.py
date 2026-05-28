import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from skimage import io, color

img = io.imread("elephant.jpg")
if img.ndim == 3:
    img = color.rgb2gray(img)
img = img.astype(np.float64)

# ════════════════════════════════════════════════════════════
# Part 1.1: Finite Difference Operator
# ════════════════════════════════════════════════════════════

Dx = np.array([[1, -1]])
Dy = np.array([[1], [-1]])

grad_x = convolve2d(img, Dx, mode='same', boundary='symm')
grad_y = convolve2d(img, Dy, mode='same', boundary='symm')
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

thresholds = [0.05, 0.1, 0.13, 0.15]

# ── Figure 1: x 方向邊緣偵測 ──────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
fig1.suptitle("X-direction Edge Detection (Dx)", fontsize=13, fontweight='bold')

vmax_x = np.abs(grad_x).max()
axes1[0].imshow(grad_x, cmap='gray', vmin=-vmax_x, vmax=vmax_x)
axes1[0].set_title("dI/dx")
axes1[0].axis('off')

axes1[1].imshow((np.abs(grad_x) > 0.1).astype(float), cmap='gray')
axes1[1].set_title("|dI/dx| binarized (threshold=0.1)")
axes1[1].axis('off')

plt.tight_layout()
plt.savefig("part1_x.png", dpi=150, bbox_inches='tight')

# ── Figure 2: y 方向邊緣偵測 ──────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.suptitle("Y-direction Edge Detection (Dy)", fontsize=13, fontweight='bold')

vmax_y = np.abs(grad_y).max()
axes2[0].imshow(grad_y, cmap='gray', vmin=-vmax_y, vmax=vmax_y)
axes2[0].set_title("dI/dy")
axes2[0].axis('off')

axes2[1].imshow((np.abs(grad_y) > 0.1).astype(float), cmap='gray')
axes2[1].set_title("|dI/dy| binarized (threshold=0.1)")
axes2[1].axis('off')

plt.tight_layout()
plt.savefig("part1_y.png", dpi=150, bbox_inches='tight')

# ── Figure 3: xy 兩方向（梯度幅度）+ 四種閾值比較 ──────────
fig3, axes3 = plt.subplots(1, 5, figsize=(22, 5))
fig3.suptitle("XY Edge Detection (Gradient Magnitude)", fontsize=13, fontweight='bold')

axes3[0].imshow(grad_magnitude, cmap='gray')
axes3[0].set_title("Gradient Magnitude |∇I|")
axes3[0].axis('off')

for i, t in enumerate(thresholds):
    edge = (grad_magnitude > t).astype(float)
    axes3[i + 1].imshow(edge, cmap='gray')
    axes3[i + 1].set_title(f"threshold = {t}")
    axes3[i + 1].axis('off')

plt.tight_layout()
plt.savefig("part1_xy.png", dpi=150, bbox_inches='tight')

plt.show()
print("Part 1.1 done")

# ════════════════════════════════════════════════════════════
# Part 1.2: Derivative of Gaussian (DoG) Filter
# ════════════════════════════════════════════════════════════

# 2D Gaussian kernel via outer product
ksize, sigma = 15, 2.0
g1d = gaussian(ksize, sigma)
g1d /= g1d.sum()
G = np.outer(g1d, g1d)

# --- Two-step: blur first, then finite difference ---
img_blur = convolve2d(img, G, mode='same', boundary='symm')
grad_x_blur = convolve2d(img_blur, Dx, mode='same', boundary='symm')
grad_y_blur = convolve2d(img_blur, Dy, mode='same', boundary='symm')
grad_mag_blur = np.sqrt(grad_x_blur**2 + grad_y_blur**2)

# --- Single-step: DoG filter = G * Dx / G * Dy ---
DoG_x = convolve2d(G, Dx, mode='same', boundary='symm')
DoG_y = convolve2d(G, Dy, mode='same', boundary='symm')
grad_x_dog = convolve2d(img, DoG_x, mode='same', boundary='symm')
grad_y_dog = convolve2d(img, DoG_y, mode='same', boundary='symm')
grad_mag_dog = np.sqrt(grad_x_dog**2 + grad_y_dog**2)

# Numerical verification
print(f"Max diff (blur+Dx vs DoG_x): {np.abs(grad_x_blur - grad_x_dog).max():.2e}")
print(f"Max diff (blur+Dy vs DoG_y): {np.abs(grad_y_blur - grad_y_dog).max():.2e}")

threshold_12 = 0.03

# ── Figure 4: Part 1.1 vs Part 1.2 comparison ─────────────
fig4, axes4 = plt.subplots(2, 3, figsize=(15, 10))
fig4.suptitle("Part 1.1 vs Part 1.2 — Effect of Gaussian Blur", fontsize=13, fontweight='bold')

vmax_xb = np.abs(grad_x_blur).max()
vmax_yb = np.abs(grad_y_blur).max()

axes4[0, 0].imshow(grad_x, cmap='gray', vmin=-vmax_x, vmax=vmax_x)
axes4[0, 0].set_title("1.1  dI/dx (no blur)")
axes4[0, 0].axis('off')
axes4[0, 1].imshow(grad_y, cmap='gray', vmin=-vmax_y, vmax=vmax_y)
axes4[0, 1].set_title("1.1  dI/dy (no blur)")
axes4[0, 1].axis('off')
axes4[0, 2].imshow((grad_magnitude > threshold_12).astype(float), cmap='gray')
axes4[0, 2].set_title(f"1.1  edges (t={threshold_12})")
axes4[0, 2].axis('off')

axes4[1, 0].imshow(grad_x_blur, cmap='gray', vmin=-vmax_xb, vmax=vmax_xb)
axes4[1, 0].set_title("1.2  dI/dx (Gaussian blur)")
axes4[1, 0].axis('off')
axes4[1, 1].imshow(grad_y_blur, cmap='gray', vmin=-vmax_yb, vmax=vmax_yb)
axes4[1, 1].set_title("1.2  dI/dy (Gaussian blur)")
axes4[1, 1].axis('off')
axes4[1, 2].imshow((grad_mag_blur > threshold_12).astype(float), cmap='gray')
axes4[1, 2].set_title(f"1.2  edges (t={threshold_12})")
axes4[1, 2].axis('off')

plt.tight_layout()
plt.savefig("part1_2_comparison.png", dpi=150, bbox_inches='tight')

# ── Figure 5: DoG filters visualization ───────────────────
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
fig5.suptitle("DoG Filters", fontsize=13, fontweight='bold')

axes5[0].imshow(G, cmap='gray')
axes5[0].set_title("Gaussian kernel G")
axes5[0].axis('off')
axes5[1].imshow(DoG_x, cmap='gray', vmin=-np.abs(DoG_x).max(), vmax=np.abs(DoG_x).max())
axes5[1].set_title("DoG_x = G * Dx")
axes5[1].axis('off')
axes5[2].imshow(DoG_y, cmap='gray', vmin=-np.abs(DoG_y).max(), vmax=np.abs(DoG_y).max())
axes5[2].set_title("DoG_y = G * Dy")
axes5[2].axis('off')

plt.tight_layout()
plt.savefig("part1_2_dog_filters.png", dpi=150, bbox_inches='tight')

# ── Figure 6: Verification — two-step vs single-step ──────
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
fig6.suptitle("Verification: Two-step (blur+diff) vs Single-step (DoG)", fontsize=13, fontweight='bold')

axes6[0].imshow((grad_mag_blur > threshold_12).astype(float), cmap='gray')
axes6[0].set_title(f"Two-step edges (t={threshold_12})")
axes6[0].axis('off')
axes6[1].imshow((grad_mag_dog > threshold_12).astype(float), cmap='gray')
axes6[1].set_title(f"Single DoG conv edges (t={threshold_12})")
axes6[1].axis('off')
diff_vis = np.abs(grad_mag_blur - grad_mag_dog)
axes6[2].imshow(diff_vis, cmap='hot')
axes6[2].set_title(f"Difference (max={diff_vis.max():.2e})")
axes6[2].axis('off')

plt.tight_layout()
plt.savefig("part1_2_verify.png", dpi=150, bbox_inches='tight')

plt.show()
print("Part 1.2 done")
