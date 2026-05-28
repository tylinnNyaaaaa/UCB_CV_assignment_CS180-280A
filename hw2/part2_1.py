import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from skimage import io

img = io.imread("dog.jpg").astype(np.float64) / 255.0

# ════════════════════════════════════════════════════════════
# Part 2.1: Unsharp Masking
# ════════════════════════════════════════════════════════════

ksize, sigma = 15, 2.0
g1d = gaussian(ksize, sigma)
g1d /= g1d.sum()
G = np.outer(g1d, g1d)

alpha = 1.5

def conv_rgb(image, kernel):
    return np.stack([
        convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
        for c in range(3)
    ], axis=2)

# Unsharp mask filter: (1 + alpha) * delta - alpha * G
impulse = np.zeros_like(G)
impulse[ksize // 2, ksize // 2] = 1.0
unsharp_filter = (1 + alpha) * impulse - alpha * G

sharpened = np.clip(conv_rgb(img, unsharp_filter), 0, 1)

# Intermediate results for visualization
low_freq  = conv_rgb(img, G)
high_freq = img - low_freq

# ── Figure 1: sharpening process ──────────────────────────
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
fig1.suptitle("Unsharp Masking — Process", fontsize=13, fontweight='bold')

axes1[0].imshow(img)
axes1[0].set_title("Original")
axes1[0].axis('off')

axes1[1].imshow(np.clip(low_freq, 0, 1))
axes1[1].set_title("Low freq (Gaussian blur)")
axes1[1].axis('off')

hf_display = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min())
axes1[2].imshow(hf_display)
axes1[2].set_title("High freq (original - blur)")
axes1[2].axis('off')

axes1[3].imshow(sharpened)
axes1[3].set_title(f"Sharpened (alpha={alpha})")
axes1[3].axis('off')

plt.tight_layout()
plt.savefig("part2_1_process.png", dpi=150, bbox_inches='tight')

# ── Figure 2: alpha comparison ─────────────────────────────
alphas = [0.5, 1.0, 1.5, 2.5]
fig2, axes2 = plt.subplots(1, len(alphas) + 1, figsize=(22, 5))
fig2.suptitle("Unsharp Masking — Alpha Comparison", fontsize=13, fontweight='bold')

axes2[0].imshow(img)
axes2[0].set_title("Original")
axes2[0].axis('off')

for i, a in enumerate(alphas):
    f = (1 + a) * impulse - a * G
    axes2[i + 1].imshow(np.clip(conv_rgb(img, f), 0, 1))
    axes2[i + 1].set_title(f"alpha = {a}")
    axes2[i + 1].axis('off')

plt.tight_layout()
plt.savefig("part2_1_alpha.png", dpi=150, bbox_inches='tight')

# ── Figure 3: unsharp mask filter visualization ────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle("Unsharp Mask Filter", fontsize=13, fontweight='bold')

axes3[0].imshow(impulse, cmap='gray')
axes3[0].set_title("Impulse (delta)")
axes3[0].axis('off')

axes3[1].imshow(G, cmap='gray')
axes3[1].set_title("Gaussian G")
axes3[1].axis('off')

vmax_uf = np.abs(unsharp_filter).max()
axes3[2].imshow(unsharp_filter, cmap='gray', vmin=-vmax_uf, vmax=vmax_uf)
axes3[2].set_title(f"Unsharp filter = (1+a)*delta - a*G  (a={alpha})")
axes3[2].axis('off')

plt.tight_layout()
plt.savefig("part2_1_filter.png", dpi=150, bbox_inches='tight')

plt.show()
print("Part 2.1 done")
