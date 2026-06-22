"""
Two-image mosaic (panorama) pipeline.

Usage:
    python mosaic.py              # interactive point-picking (default)
    python mosaic.py --hardcoded  # use pre-saved correspondences
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread as mpl_imread
from homography import computeH, warpImage

matplotlib.use('TkAgg')

# ---------------------------------------------------------------------------
# Hardcoded correspondences for --hardcoded mode.
# Approximate pixel coords; use interactive mode for best results.
# Order: TV corners (TL,TR,BL,BR), clock, cabinet edge, glass-cabinet (top,bot)
# ---------------------------------------------------------------------------
IM1_PTS = np.array([
    [ 620,  280],
    [1044,  280],
    [ 620,  574],
    [1044,  574],
    [1024,  655],
    [1044,  350],
    [ 587,  375],
    [ 587,  622],
], dtype=np.float64)

IM2_PTS = np.array([
    [ 218,  168],
    [ 647,  168],
    [ 218,  452],
    [ 647,  452],
    [ 640,  545],
    [ 647,  285],
    [ 140,  264],
    [ 140,  487],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Point correspondence picker
# ---------------------------------------------------------------------------
def pick_correspondences(im1: np.ndarray, im2: np.ndarray,
                         n: int = 8) -> tuple:
    """
    Display im1 and im2 side by side in a single axes.
    Collect 2*n clicks via ginput: first n on im1, last n on im2.
    Subtract im2 x-offset (w1) before returning.
    Returns (im1_pts, im2_pts) as (n, 2) float64 arrays.
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    H_pad = max(h1, h2)

    canvas = np.zeros((H_pad, w1 + w2, 3), dtype=np.float64)
    canvas[:h1, :w1]      = im1
    canvas[:h2, w1:w1+w2] = im2

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.imshow(canvas)
    ax.axvline(x=w1 - 0.5, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax.set_title(
        f"Click {n} points in the LEFT image (im1),\n"
        f"then {n} corresponding points in the RIGHT image (im2).",
        fontsize=11,
    )
    ax.axis('off')
    plt.tight_layout()

    all_pts = np.array(plt.ginput(2 * n, timeout=300), dtype=np.float64)
    plt.close(fig)

    im1_pts = all_pts[:n].copy()
    im2_pts = all_pts[n:].copy()
    im2_pts[:, 0] -= w1   # remove im2 x-offset in the combined canvas

    return im1_pts, im2_pts


# ---------------------------------------------------------------------------
# 2. Alpha weight map
# ---------------------------------------------------------------------------
def make_alpha(shape: tuple) -> np.ndarray:
    """
    Center-weighted alpha: 1 at image center, 0 at all four edges.
    alpha[y, x] = min(x, W-1-x, y, H-1-y) / max(min(W,H)/2, 1), clipped [0,1].
    """
    H, W = shape
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    dist = np.minimum(
        np.minimum(xs[np.newaxis, :], W - 1 - xs[np.newaxis, :]),
        np.minimum(ys[:, np.newaxis], H - 1 - ys[:, np.newaxis]),
    )
    return np.clip(dist / max(min(W, H) / 2.0, 1.0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Canvas construction
# ---------------------------------------------------------------------------
def build_canvas(im1: np.ndarray, im2: np.ndarray, H: np.ndarray) -> tuple:
    """
    H maps im2 → im1 (pass computeH(im2_pts, im1_pts)).
    warpImage(im2, H) places warped_im2 so its canvas origin is (x_min, y_min)
    of im2's mapped bounding box.  im1 lives at (0,0) in target space, so its
    top-left on the canvas is at (x_off, y_off) = (max(0,-x_min), max(0,-y_min)).
    Canvas is sized large enough to contain both images.

    Returns:
        canvas_im1, canvas_im2       : (H_c, W_c, C) float64
        canvas_alpha1, canvas_alpha2 : (H_c, W_c) float64
        x_offset, y_offset           : int  (where im1's top-left sits in canvas)
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    C = im1.shape[2]

    # Forward-map im2 corners through H to find im2's footprint in im1 space
    corners_im2 = np.array(
        [[0, 0, 1], [w2, 0, 1], [0, h2, 1], [w2, h2, 1]], dtype=np.float64
    ).T
    proj = H @ corners_im2
    proj /= proj[2:3]
    x_min, x_max_2 = proj[0].min(), proj[0].max()
    y_min, y_max_2 = proj[1].min(), proj[1].max()

    # im1's top-left in canvas coords: (max(0,-x_min), max(0,-y_min))
    x_off = max(0, int(round(-x_min)))
    y_off = max(0, int(round(-y_min)))

    # Canvas covers both im1 [0,w1]×[0,h1] and im2 [x_min,x_max_2]×[y_min,y_max_2]
    W_c = int(np.ceil(max(float(w1), x_max_2) - min(0.0, x_min)))
    H_c = int(np.ceil(max(float(h1), y_max_2) - min(0.0, y_min)))

    # --- Place im1 ---
    canvas_im1 = np.zeros((H_c, W_c, C), dtype=np.float64)
    canvas_im1[y_off:y_off + h1, x_off:x_off + w1] = im1

    canvas_alpha1 = np.zeros((H_c, W_c), dtype=np.float64)
    canvas_alpha1[y_off:y_off + h1, x_off:x_off + w1] = make_alpha((h1, w1))

    # --- Warp im2 and place ---
    # warped_im2[0,0] = im1-space (x_min, y_min) = canvas (x_min+x_off, y_min+y_off)
    warped_im2, _ = warpImage(im2, H)
    h2w, w2w = warped_im2.shape[:2]

    r0 = max(0, int(round(y_min + y_off)))
    c0 = max(0, int(round(x_min + x_off)))
    r1 = min(r0 + h2w, H_c)
    c1 = min(c0 + w2w, W_c)
    dr, dc = r1 - r0, c1 - c0

    canvas_im2 = np.zeros((H_c, W_c, C), dtype=np.float64)
    canvas_im2[r0:r1, c0:c1] = warped_im2[:dr, :dc]

    # Warp im2's alpha through the same H
    alpha2_src = make_alpha((h2, w2))[:, :, np.newaxis]   # (h2, w2, 1)
    warped_alpha2, _ = warpImage(alpha2_src, H)            # (h2w, w2w, 1)

    canvas_alpha2 = np.zeros((H_c, W_c), dtype=np.float64)
    canvas_alpha2[r0:r1, c0:c1] = warped_alpha2[:dr, :dc, 0]

    return canvas_im1, canvas_im2, canvas_alpha1, canvas_alpha2, x_off, y_off


# ---------------------------------------------------------------------------
# 4. Blending
# ---------------------------------------------------------------------------
def blend(canvas_im1: np.ndarray, canvas_im2: np.ndarray,
          alpha1: np.ndarray, alpha2: np.ndarray) -> np.ndarray:
    """Weighted average: (a1*im1 + a2*im2) / (a1 + a2 + eps), clipped [0,1]."""
    a1 = alpha1[:, :, np.newaxis]
    a2 = alpha2[:, :, np.newaxis]
    return np.clip(
        (a1 * canvas_im1 + a2 * canvas_im2) / (a1 + a2 + 1e-8),
        0.0, 1.0,
    )


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    hardcoded = "--hardcoded" in sys.argv

    im1 = mpl_imread("img1.jpg").astype(np.float64) / 255.0
    im2 = mpl_imread("img2.jpg").astype(np.float64) / 255.0

    npts = 8
    if "--npts" in sys.argv:
        npts = int(sys.argv[sys.argv.index("--npts") + 1])

    if hardcoded:
        im1_pts, im2_pts = IM1_PTS.copy(), IM2_PTS.copy()
        print(f"Using hardcoded correspondences ({len(IM1_PTS)} pairs).")
    else:
        print(f"Interactive mode ({npts} point pairs):")
        print("  Click all points in im1 (left panel) first,")
        print("  then the corresponding points in im2 (right panel).")
        im1_pts, im2_pts = pick_correspondences(im1, im2, n=npts)

    n = len(im1_pts)
    print(f"\nim1_pts =\n{im1_pts}")
    print(f"\nim2_pts =\n{im2_pts}")

    # H maps im2 → im1 so warpImage(im2, H) puts im2 into im1's coordinate frame
    H = computeH(im2_pts, im1_pts)
    print(f"\nH (im2→im1, {n} pairs, least-squares) =\n{H}")

    canvas_im1, canvas_im2, alpha1, alpha2, x_off, y_off = build_canvas(im1, im2, H)
    mosaic = blend(canvas_im1, canvas_im2, alpha1, alpha2)

    print(f"\nCanvas : {mosaic.shape[1]} × {mosaic.shape[0]}")
    print(f"im1 offset in canvas : x={x_off}, y={y_off}")

    # --- Visualise ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Two-Image Mosaic", fontsize=14, fontweight="bold")

    axes[0].imshow(im1)
    axes[0].set_title(f"im1  ({im1.shape[1]}×{im1.shape[0]})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(im2)
    axes[1].set_title(f"im2  ({im2.shape[1]}×{im2.shape[0]})", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(mosaic)
    axes[2].set_title(f"Mosaic  ({mosaic.shape[1]}×{mosaic.shape[0]})", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = "mosaic_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
