import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import io
from skimage.transform import resize


# ── Stack helpers ──────────────────────────────────────────

def make_gaussian(ksize, sigma):
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-ax**2 / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def conv_img(img, kernel):
    if img.ndim == 2:
        return convolve2d(img, kernel, mode='same', boundary='symm')
    return np.stack([
        convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        for c in range(img.shape[2])
    ], axis=2)


def build_gaussian_stack(img, sigma, n_levels):
    ksize = int(6 * sigma + 1) | 1     # ensure odd
    G = make_gaussian(ksize, sigma)
    stack = [img]
    for _ in range(n_levels):
        stack.append(conv_img(stack[-1], G))
    return stack


def build_laplacian_stack(g_stack):
    l_stack = [g_stack[k] - g_stack[k + 1] for k in range(len(g_stack) - 1)]
    l_stack.append(g_stack[-1])        # residual
    return l_stack


# ── Core blend ─────────────────────────────────────────────

def multiresolution_blend(A, B, mask, sigma, n_levels):
    """
    A, B : (H, W) or (H, W, 3) in [0, 1]
    mask : (H, W) in [0, 1]  — 1 = show A, 0 = show B
    Returns blended image clipped to [0, 1].
    """
    g_A    = build_gaussian_stack(A,    sigma, n_levels)
    g_B    = build_gaussian_stack(B,    sigma, n_levels)
    g_mask = build_gaussian_stack(mask, sigma, n_levels)

    l_A = build_laplacian_stack(g_A)
    l_B = build_laplacian_stack(g_B)

    blended_levels = []
    for i in range(len(l_A)):
        m = g_mask[i]
        if A.ndim == 3:
            m = m[:, :, np.newaxis]    # broadcast over RGB channels
        blended_levels.append(l_A[i] * m + l_B[i] * (1 - m))

    return np.clip(np.sum(blended_levels, axis=0), 0, 1), l_A, l_B, g_mask


def normalize(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    sigma      = 4.0
    n_levels   = 8
    mask_sigma = 20.0   # pre-blur strength for the mask seam

    A = io.imread("apple.jpg").astype(np.float64)  / 255.0
    B = io.imread("orange.jpg").astype(np.float64) / 255.0
    B = resize(B, A.shape, anti_aliasing=True)

    # vertical seam mask: left half = A, right half = B
    mask = np.zeros(A.shape[:2])
    mask[:, : A.shape[1] // 2] = 1.0

    # pre-blur the mask so the seam is already softened before the stack
    mask_ksize = int(6 * mask_sigma + 1) | 1
    G_mask = make_gaussian(mask_ksize, mask_sigma)
    mask = convolve2d(mask, G_mask, mode='same', boundary='symm')

    oraple, l_A, l_B, g_mask = multiresolution_blend(A, B, mask, sigma, n_levels)

    print(f"oraple  min={oraple.min():.3f}  max={oraple.max():.3f}")

    # ── Figure 1: result overview ──────────────────────────
    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
    fig1.suptitle("Multiresolution Blending — Oraple", fontweight='bold')
    for ax, im, t in zip(axes1,
                         [A, B, mask, oraple],
                         ["Apple (A)", "Orange (B)", "Mask", "Oraple (blended)"]):
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("part2_4_oraple_sigma4.png", dpi=150, bbox_inches='tight')

    # ── Figure 2: Figure-10 style process (levels 0,2,4 + result) ──
    show_levels = [0, 2, 4, n_levels]   # last = residual level
    n_show = len(show_levels)

    fig2, axes2 = plt.subplots(3, n_show, figsize=(5 * n_show, 15))
    fig2.suptitle("Laplacian Stack Blending Process (Fig. 10 style)", fontweight='bold')

    for col, lvl in enumerate(show_levels):
        label = f"L{lvl}" if lvl < n_levels else "residual"
        m = g_mask[lvl][:, :, np.newaxis] if A.ndim == 3 else g_mask[lvl]

        la_masked = normalize(l_A[lvl] * m)
        lb_masked = normalize(l_B[lvl] * (1 - m))
        blended_lvl = normalize(l_A[lvl] * m + l_B[lvl] * (1 - m))

        axes2[0, col].imshow(la_masked, cmap=None)
        axes2[0, col].set_title(f"L_A[{label}] × M")
        axes2[0, col].axis('off')

        axes2[1, col].imshow(lb_masked, cmap=None)
        axes2[1, col].set_title(f"L_B[{label}] × (1−M)")
        axes2[1, col].axis('off')

        axes2[2, col].imshow(blended_lvl, cmap=None)
        axes2[2, col].set_title(f"Blended [{label}]")
        axes2[2, col].axis('off')

    plt.tight_layout()
    plt.savefig("part2_4_process_sigma4.png", dpi=150, bbox_inches='tight')

    plt.show()
    print("Part 2.4 done")
