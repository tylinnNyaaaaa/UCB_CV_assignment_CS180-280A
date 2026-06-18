"""
CS180 Project 3 — Bells & Whistles: Non-linear Pixel Distortion (Liquify)
Implements bloat / shrink / twirl via inverse pixel mapping.

Usage: python part5_liquify.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part12 import HW3, OUT
from part4 import parse_asf, DATA_DIR


# ── Inverse-mapping helpers ───────────────────────────────────────────────────

def _sample(img, src_rows, src_cols):
    """Bilinear-interpolate img at (src_rows, src_cols), per channel."""
    coords = [src_rows, src_cols]
    if img.ndim == 3:
        out = np.stack([
            map_coordinates(img[:, :, c].astype(float), coords,
                            order=1, mode='nearest')
            for c in range(img.shape[2])
        ], axis=-1)
    else:
        out = map_coordinates(img.astype(float), coords, order=1, mode='nearest')
    return np.clip(out, 0, 255).astype(np.uint8)


def _grid(img):
    h, w = img.shape[:2]
    return np.mgrid[0:h, 0:w].astype(float)   # (2, H, W): [rows, cols]


def _falloff(rows, cols, cy, cx, radius):
    """Gaussian falloff mask centred at (cx, cy)."""
    dy, dx = rows - cy, cols - cx
    return np.exp(-((dx**2 + dy**2) / radius**2)), dx, dy


# ── Three distortion modes ────────────────────────────────────────────────────

def bloat(img, cx, cy, radius, strength=0.5):
    """
    Forward: push pixels away from (cx, cy)  →  features look enlarged.
    Inverse map: output pixel ← source closer to centre.
    """
    rows, cols = _grid(img)
    mask, dx, dy = _falloff(rows, cols, cy, cx, radius)
    scale = np.maximum(1 - strength * mask, 0.01)
    return _sample(img, cy + dy * scale, cx + dx * scale)


def shrink(img, cx, cy, radius, strength=0.5):
    """
    Forward: pull pixels toward (cx, cy)  →  features look compressed.
    Inverse map: output pixel ← source farther from centre.
    """
    rows, cols = _grid(img)
    mask, dx, dy = _falloff(rows, cols, cy, cx, radius)
    scale = 1 + strength * mask
    return _sample(img, cy + dy * scale, cx + dx * scale)


def twirl(img, cx, cy, radius, strength=1.5):
    """
    Rotates pixels around (cx, cy); rotation angle decays with distance.
    Inverse map: rotate in the opposite direction.
    """
    rows, cols = _grid(img)
    mask, dx, dy = _falloff(rows, cols, cy, cx, radius)
    angle = -strength * mask          # inverse direction
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    src_cols = cx + cos_a * dx - sin_a * dy
    src_rows = cy + sin_a * dx + cos_a * dy
    return _sample(img, src_rows, src_cols)


# ── Demo on the female IMM face ───────────────────────────────────────────────

def keypoint_centers(pts):
    """
    IMM 58-pt convention:
      0-12  jaw,  13-20 right eye,  21-28 left eye,
      29-33 right brow, 34-38 left brow,
      39-46 mouth,  47-57 nose
    Returns dict of (cx, cy) centres.
    """
    return {
        'right_eye': pts[13:21].mean(axis=0),
        'left_eye':  pts[21:29].mean(axis=0),
        'mouth':     pts[39:47].mean(axis=0),
        'nose':      pts[47:58].mean(axis=0),
        'face':      pts[0:13].mean(axis=0),
    }


def save_grid(original, results, titles, suptitle, path):
    n = 1 + len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    axes[0].imshow(original);  axes[0].set_title('Original');  axes[0].axis('off')
    for ax, img, title in zip(axes[1:], results, titles):
        ax.imshow(img);  ax.set_title(title);  ax.axis('off')
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def main():
    img_path = os.path.join(DATA_DIR, '08-1f.jpg')
    asf_path = os.path.join(DATA_DIR, '08-1f.asf')

    img = np.array(Image.open(img_path).convert('RGB'))
    pts = parse_asf(asf_path)          # (58, 2) in (x, y) pixel coords
    kp  = keypoint_centers(pts)

    re_cx, re_cy = kp['right_eye']
    le_cx, le_cy = kp['left_eye']
    fc_cx, fc_cy = kp['face']
    mo_cx, mo_cy = kp['mouth']
    no_cx, no_cy = kp['nose']

    # ── Bloat demo: enlarge both eyes ────────────────────────────────────────
    r_eye = 55          # radius of effect in pixels
    bloat_r = bloat(img,  re_cx, re_cy, radius=r_eye, strength=0.6)
    bloat_both = bloat(bloat_r, le_cx, le_cy, radius=r_eye, strength=0.6)

    # ── Shrink demo: slim the jaw ─────────────────────────────────────────────
    slim = shrink(img, fc_cx, fc_cy, radius=180, strength=0.35)

    # ── Twirl demo: different strengths ──────────────────────────────────────
    twirl_mild   = twirl(img, no_cx, no_cy, radius=120, strength=1.0)
    twirl_strong = twirl(img, no_cx, no_cy, radius=120, strength=2.5)

    # ── Save grids ────────────────────────────────────────────────────────────
    save_grid(
        img,
        [bloat_both, slim],
        ['Bloat eyes (r=55, s=0.6)', 'Shrink jaw (r=180, s=0.35)'],
        'Bloat & Shrink',
        os.path.join(OUT, 'part5_liquify_bloat_shrink.png'),
    )

    save_grid(
        img,
        [twirl_mild, twirl_strong],
        ['Twirl mild (s=1.0)', 'Twirl strong (s=2.5)'],
        'Twirl',
        os.path.join(OUT, 'part5_liquify_twirl.png'),
    )


if __name__ == '__main__':
    main()