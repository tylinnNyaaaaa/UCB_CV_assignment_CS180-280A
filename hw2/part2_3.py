import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io


def make_gaussian(ksize, sigma):
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-ax**2 / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def conv_img(img, kernel):
    """Convolve grayscale or RGB image with kernel."""
    if img.ndim == 2:
        return convolve2d(img, kernel, mode='same', boundary='symm')
    return np.stack([
        convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        for c in range(img.shape[2])
    ], axis=2)


def build_gaussian_stack(img, sigma, n_levels):
    """
    Level 0 = original. Each subsequent level applies one more Gaussian blur.
    Returns list of length n_levels + 1.
    """
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    G = make_gaussian(ksize, sigma)

    stack = [img]
    for _ in range(n_levels):
        stack.append(conv_img(stack[-1], G))
    return stack


def build_laplacian_stack(g_stack):
    """
    Level k = g_stack[k] - g_stack[k+1].
    Last level = g_stack[-1] (low-frequency residual).
    Returns list of same length as g_stack.
    """
    l_stack = [g_stack[k] - g_stack[k + 1] for k in range(len(g_stack) - 1)]
    l_stack.append(g_stack[-1])
    return l_stack


def normalize(img):
    """Shift to [0, 1] for display of signed Laplacian levels."""
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def show_stack(axes, stack, title_prefix, normalize_fn=None):
    for i, (ax, img) in enumerate(zip(axes, stack)):
        disp = normalize_fn(img) if normalize_fn else np.clip(img, 0, 1)
        ax.imshow(disp, cmap='gray' if disp.ndim == 2 else None)
        ax.set_title(f"{title_prefix} L{i}")
        ax.axis('off')


if __name__ == "__main__":
    sigma   = 2.0
    n_levels = 5      # levels 0–5: 6 images per stack

    img = io.imread("apple.jpg").astype(np.float64) / 255.0

    g_stack = build_gaussian_stack(img, sigma, n_levels)
    l_stack = build_laplacian_stack(g_stack)

    # reconstruction sanity check
    reconstructed = np.sum(l_stack, axis=0)
    print(f"Reconstruction error (max): {np.abs(reconstructed - img).max():.2e}")

    n = len(g_stack)   # n_levels + 1

    # ── Figure 1: Gaussian stack ───────────────────────────────
    fig1, axes1 = plt.subplots(1, n, figsize=(4 * n, 4))
    fig1.suptitle(f"Gaussian Stack  (sigma={sigma}, {n_levels} levels)", fontweight='bold')
    show_stack(axes1, g_stack, "G")
    plt.tight_layout()
    plt.savefig("part2_3_gaussian.png", dpi=150, bbox_inches='tight')

    # ── Figure 2: Laplacian stack ──────────────────────────────
    fig2, axes2 = plt.subplots(1, n, figsize=(4 * n, 4))
    fig2.suptitle(f"Laplacian Stack  (last level = residual)", fontweight='bold')
    show_stack(axes2, l_stack, "L", normalize_fn=normalize)
    plt.tight_layout()
    plt.savefig("part2_3_laplacian.png", dpi=150, bbox_inches='tight')

    # ── Figure 3: Reconstruction verification ─────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("Reconstruction from Laplacian Stack", fontweight='bold')
    axes3[0].imshow(img, cmap='gray');          axes3[0].set_title("Original");        axes3[0].axis('off')
    axes3[1].imshow(np.clip(reconstructed, 0, 1), cmap='gray')
    axes3[1].set_title("Reconstructed");        axes3[1].axis('off')
    diff = np.abs(reconstructed - img)
    axes3[2].imshow(diff, cmap='hot');          axes3[2].set_title(f"Diff (max={diff.max():.2e})"); axes3[2].axis('off')
    plt.tight_layout()
    plt.savefig("part2_3_reconstruct.png", dpi=150, bbox_inches='tight')

    plt.show()
    print("Part 2.3 done")
