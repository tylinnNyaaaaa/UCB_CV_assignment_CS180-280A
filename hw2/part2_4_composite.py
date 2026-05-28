import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import binary_fill_holes
from skimage import io, morphology
from skimage.transform import resize


# ── Core helpers ───────────────────────────────────────────

def make_gaussian(ksize, sigma):
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-ax**2 / (2 * sigma**2))
    g /= g.sum()
    return np.outer(g, g)


def conv_rgb(img, kernel):
    if img.ndim == 2:
        return convolve2d(img, kernel, mode='same', boundary='symm')
    return np.stack([
        convolve2d(img[:, :, c], kernel, mode='same', boundary='symm')
        for c in range(img.shape[2])
    ], axis=2)


def low_pass(img, sigma, ksize=None):
    if ksize is None:
        ksize = int(6 * sigma + 1) | 1
    return conv_rgb(img, make_gaussian(ksize, sigma))


def gaussian_stack(img, levels, sigma):
    stack = [img]
    for _ in range(levels):
        stack.append(low_pass(stack[-1], sigma))
    return stack


def laplacian_stack(img, levels, sigma):
    g = gaussian_stack(img, levels, sigma)
    l = [g[k] - g[k + 1] for k in range(levels)]
    l.append(g[-1])
    return l


def blend(IA, IB, mask, levels=6, sigma=2):
    g_A    = gaussian_stack(IA,   levels, sigma)
    g_B    = gaussian_stack(IB,   levels, sigma)
    g_mask = gaussian_stack(mask, levels, sigma)
    l_A    = laplacian_stack(IA,  levels, sigma)
    l_B    = laplacian_stack(IB,  levels, sigma)

    blended = []
    for i in range(levels + 1):
        m = g_mask[i]
        if IA.ndim == 3:
            m = m[:, :, np.newaxis]
        blended.append(l_A[i] * m + l_B[i] * (1 - m))
    return np.clip(np.sum(blended, axis=0), 0, 1), g_mask


# ── Mask helpers ───────────────────────────────────────────

def make_mask(B_rolled, white_threshold=0.95):
    """
    mask = 1 where B is subject (non-white)
    mask = 0 where B is white background or zero-padded
    Invert before passing to blend() so that mask=1 → show A, mask=0 → show B.
    """
    zero_region = B_rolled.sum(axis=2) == 0
    is_white    = np.all(B_rolled >= white_threshold, axis=2)
    mask = (~is_white).astype(float)
    mask[zero_region] = 0
    return mask


def clean_mask(mask):
    m = binary_fill_holes(mask.astype(bool))
    m = morphology.remove_small_objects(m, max_size=500)
    m = morphology.remove_small_holes(m, max_size=500)
    return m.astype(np.float64)


def normalize(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────
    A_path     = "beach.jpeg"
    B_path     = "kabi.jpg"
    dy, dx          = 50, 300  # roll offsets: dy=down, dx=right
    scale_b         = 0.75     # extra scale for B (1.0 = fit-to-A, 0.5 = half that)
    white_threshold = 0.95
    levels          = 6
    sigma           = 2

    # ── Step 1: load and align ────────────────────────────
    A = io.imread(A_path).astype(np.float64) / 255.0
    B = io.imread(B_path).astype(np.float64) / 255.0
    H, W = A.shape[:2]

    # resize B: first fit within A, then apply extra scale_b
    bh, bw = B.shape[:2]
    base_scale = min(H / bh, W / bw)
    final_scale = base_scale * scale_b
    B = resize(B, (max(1, int(bh * final_scale)), max(1, int(bw * final_scale))), anti_aliasing=True)
    bh, bw = B.shape[:2]

    # zero-pad B to A's dimensions
    B_padded = np.zeros_like(A)
    B_padded[:bh, :bw] = B

    # roll to position
    B_rolled = np.roll(B_padded, shift=(dy, dx), axis=(0, 1))

    # ── Step 2: auto mask ─────────────────────────────────
    mask_raw   = make_mask(B_rolled, white_threshold=white_threshold)
    mask_clean = clean_mask(mask_raw)

    # ── Step 3: blend ─────────────────────────────────────
    # invert: blend() uses mask=1→A, mask=0→B; clean mask has 1=subject
    A_float    = A
    mask_blend = 1.0 - mask_clean
    result, g_mask = blend(A_float, B_rolled, mask_blend, levels=levels, sigma=sigma)

    # ── Figure 1: alignment check (1×2) ───────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle("Alignment Check", fontweight='bold')
    axes1[0].imshow(A_float);                            axes1[0].set_title("Scene A");       axes1[0].axis('off')
    axes1[1].imshow(np.clip(A_float + B_rolled, 0, 1)); axes1[1].set_title("A + B overlay"); axes1[1].axis('off')
    plt.tight_layout()
    plt.savefig("part2_4_align.png", dpi=150, bbox_inches='tight')

    # ── Figure 2: mask at different thresholds (1×3) ──────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("Mask — White Threshold Comparison", fontweight='bold')
    for ax, t in zip(axes2, [0.75, 0.85, 0.95]):
        raw = make_mask(B_rolled, white_threshold=t)
        ax.imshow(raw, cmap='gray')
        ax.set_title(f"white_threshold={t}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("part2_4_masks.png", dpi=150, bbox_inches='tight')

    # ── Figure 3: compositing result (1×4) ────────────────
    fig3, axes3 = plt.subplots(1, 4, figsize=(22, 6))
    fig3.suptitle("Compositing Result", fontweight='bold')
    for ax, im, t in zip(axes3,
                         [A_float, B_rolled, mask_clean, result],
                         ["Scene A", "Subject B (positioned)", "Mask (cleaned)", "Result"]):
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("part2_4_result.png", dpi=150, bbox_inches='tight')

    # ── Figure 4: blending process (2 rows) ───────────────
    l_result = laplacian_stack(result, levels, sigma)
    n_cols   = levels + 1

    fig4, axes4 = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    fig4.suptitle("Blending Process — Laplacian stack of result & Mask Gaussian stack", fontweight='bold')
    for col in range(n_cols):
        label = f"L{col}" if col < levels else "residual"
        axes4[0, col].imshow(normalize(l_result[col]))
        axes4[0, col].set_title(f"result {label}")
        axes4[0, col].axis('off')
        axes4[1, col].imshow(g_mask[col], cmap='gray')
        axes4[1, col].set_title(f"mask G{col}")
        axes4[1, col].axis('off')
    plt.tight_layout()
    plt.savefig("part2_4_process.png", dpi=150, bbox_inches='tight')

    plt.show()
    print("Part 2.4 composite done")
