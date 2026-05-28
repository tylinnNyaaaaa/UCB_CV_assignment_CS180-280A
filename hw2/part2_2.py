import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io, color
from skimage.transform import rescale, resize


def make_gaussian(ksize, sigma):
    """Return normalized 2D Gaussian kernel (ksize x ksize)."""
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-ax**2 / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def conv_rgb(img, kernel):
    """Convolve each RGB channel independently. mode='same', boundary='symm'."""
    if img.ndim == 2:
        return convolve2d(img, kernel, mode='same', boundary='symm')
    return np.stack([
        convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        for c in range(img.shape[2])
    ], axis=2)


def low_pass(img, sigma, ksize=None):
    """ksize defaults to 6*sigma+1 (rounded to odd)."""
    if ksize is None:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
    return conv_rgb(img, make_gaussian(ksize, sigma))


def high_pass(img, sigma, ksize=None):
    """Return img - low_pass(img)."""
    return img - low_pass(img, sigma, ksize)


def make_hybrid(I1, I2, sigma1, sigma2):
    """Returns np.clip(low_pass(I1) + high_pass(I2), 0, 1). Supports grayscale and RGB."""
    return np.clip(low_pass(I1, sigma1) + high_pass(I2, sigma2), 0, 1)


def log_fft(img):
    gray = color.rgb2gray(img) if img.ndim == 3 else img
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


if __name__ == "__main__":
    # config
    I1_path = "harry.jpeg"
    I2_path  = "cat.jpg"
    sigma1  = 5     # low-pass cutoff for I1
    sigma2  = 10    # high-pass cutoff for I2

    # load & resize I2 to match I1
    I1 = io.imread(I1_path).astype(np.float64) / 255.0
    I2 = io.imread(I2_path).astype(np.float64) / 255.0
    I2 = resize(I2, I1.shape, anti_aliasing=True)

    lp_I1   = np.clip(low_pass(I1, sigma1), 0, 1)
    hp_I2   = high_pass(I2, sigma2)
    hybrid  = make_hybrid(I1, I2, sigma1, sigma2)
    hp_disp = (hp_I2 - hp_I2.min()) / (hp_I2.max() - hp_I2.min())

    print(f"hybrid min={hybrid.min():.3f}  max={hybrid.max():.3f}")

    images = [I1, lp_I1, I2, hp_disp, hybrid]
    titles = ["I1 (harry)", "low_pass(I1)", "I2 (cat)", "high_pass(I2)", "Hybrid"]

    # ── Figure 1: process breakdown (1x5) ─────────────────────
    fig1, axes1 = plt.subplots(1, 5, figsize=(25, 5))
    fig1.suptitle("Hybrid Image — Process Breakdown", fontsize=13, fontweight='bold')
    for ax, im, title in zip(axes1, images, titles):
        ax.imshow(im, cmap=None)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("part2_2_process.png", dpi=150, bbox_inches='tight')

    # ── Figure 2: Fourier analysis (2x5) ──────────────────────
    fig2, axes2 = plt.subplots(2, 5, figsize=(25, 10))
    fig2.suptitle("Fourier Analysis", fontsize=13, fontweight='bold')
    for col, (im, title) in enumerate(zip(images, titles)):
        axes2[0, col].imshow(im)
        axes2[0, col].set_title(title)
        axes2[0, col].axis('off')
        axes2[1, col].imshow(log_fft(im), cmap='magma')
        axes2[1, col].set_title(f"log FFT")
        axes2[1, col].axis('off')
    plt.tight_layout()
    plt.savefig("part2_2_fourier.png", dpi=150, bbox_inches='tight')

    # ── Figure 3: distance simulation (1x4) ───────────────────
    scales = [1.0, 0.5, 0.25, 0.125]
    labels = ["close", "medium", "far", "very far"]

    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
    fig3.suptitle("Distance Simulation", fontsize=13, fontweight='bold')
    for ax, scale, label in zip(axes3, scales, labels):
        if scale == 1.0:
            display = hybrid
        else:
            small   = rescale(hybrid, scale, channel_axis=-1, anti_aliasing=True)
            display = resize(small, hybrid.shape, anti_aliasing=True)
        ax.imshow(np.clip(display, 0, 1))
        ax.set_title(f"{label}  (x{scale})")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("part2_2_distance.png", dpi=150, bbox_inches='tight')

    plt.show()
    print("Part 2.2 done")
