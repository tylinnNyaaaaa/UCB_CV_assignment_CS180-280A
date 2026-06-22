"""
Cylindrical Panorama — direct inverse sampling from originals.

For each output pixel (u, v) in the cylindrical canvas, compute the 3-D ray,
back-project it into whichever original image covers that angle, and sample
directly — no intermediate re-projection step, maximum resolution.

Math recap (cylindrical coords centered at image center):
    θ = (u − cx) / f
    x_src = f · tan(θ) + cx
    y_src = (v − cy) / cos(θ) + cy

Usage:
    python project_cylinder.py                  # interactive (default 8 pts)
    python project_cylinder.py --npts 16        # pick more correspondences
    python project_cylinder.py --hardcoded      # saved correspondences
    python project_cylinder.py --focal 1400     # override focal length
"""

import sys
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.image import imread as mpl_imread
from homography import computeH, warpImage

# ---------------------------------------------------------------------------
# Hardcoded correspondences (img1 pixels ↔ img2 pixels)
# ---------------------------------------------------------------------------
IM1_PTS = np.array([
    [ 620,  280],   # TV top-left
    [1044,  280],   # TV top-right
    [ 620,  574],   # TV bottom-left
    [1044,  574],   # TV bottom-right
    [1024,  655],   # digital clock
    [1044,  350],   # cabinet right edge
    [ 587,  375],   # glass cabinet top-right
    [ 587,  622],   # glass cabinet bottom-right
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

DEFAULT_FOCAL = 1200.0   # pixels — good starting point for ~60° horizontal FOV


# ---------------------------------------------------------------------------
# Core: cylindrical inverse sampling
# ---------------------------------------------------------------------------
def project_to_cylinder(im: np.ndarray, f: float) -> tuple:
    """
    Inverse-sample im onto a cylinder of focal-length radius f.

    For each output pixel (u, v) (same grid size as im):
        θ      = (u − cx) / f
        x_src  = f · tan(θ) + cx          ← back-projected image column
        y_src  = (v − cy) / cos(θ) + cy   ← back-projected image row

    Returns:
        cyl_im : (H, W, C) float64  — cylindrical image
        valid  : (H, W) bool        — True where source pixel was in-bounds
    """
    H_im, W_im = im.shape[:2]
    cx, cy = W_im / 2.0, H_im / 2.0
    C = im.shape[2]

    U, V = np.meshgrid(np.arange(W_im, dtype=np.float64),
                       np.arange(H_im, dtype=np.float64))

    theta = (U - cx) / f
    cos_t = np.cos(theta)

    x_src = f * np.tan(theta) + cx
    y_src = (V - cy) / cos_t + cy

    valid = (x_src >= 0) & (x_src < W_im) & (y_src >= 0) & (y_src < H_im)

    N = H_im * W_im
    sampled = map_coordinates(
        im,
        [np.tile(y_src.ravel(), C),
         np.tile(x_src.ravel(), C),
         np.repeat(np.arange(C), N)],
        order=1, mode='constant', cval=0.0,
    )
    cyl_im = sampled.reshape(C, H_im, W_im).transpose(1, 2, 0)
    cyl_im[~valid] = 0.0
    return cyl_im, valid


# ---------------------------------------------------------------------------
# Offset estimation
# ---------------------------------------------------------------------------
def to_cylinder_uv(pts: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """Map image (x, y) → cylindrical (u, v)."""
    theta = np.arctan((pts[:, 0] - cx) / f)
    cos_t = f / np.sqrt((pts[:, 0] - cx) ** 2 + f ** 2)
    u = cx + f * theta
    v = cy + (pts[:, 1] - cy) * cos_t
    return np.stack([u, v], axis=1)


def estimate_offset(im1_pts: np.ndarray, im2_pts: np.ndarray,
                    f: float, W: int, H: int) -> tuple:
    """
    Estimate the panorama offset (du, dv) between the two cylindrical images.

    A feature at cyl-coord u1 in cyl1 and u2 in cyl2 must land at the same
    canvas column, so:
        c1_off + u1 = c2_off + u2  →  c2_off − c1_off = u1 − u2 = du

    Returns (du, dv):  positive du means cyl2 is to the RIGHT of cyl1.
    """
    cx, cy = W / 2.0, H / 2.0
    uv1 = to_cylinder_uv(im1_pts, f, cx, cy)
    uv2 = to_cylinder_uv(im2_pts, f, cx, cy)
    du = float(np.median(uv1[:, 0] - uv2[:, 0]))
    dv = float(np.median(uv1[:, 1] - uv2[:, 1]))
    return du, dv


# ---------------------------------------------------------------------------
# Alpha weight map (re-used from mosaic.py)
# ---------------------------------------------------------------------------
def make_alpha(shape: tuple) -> np.ndarray:
    H, W = shape
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    dist = np.minimum(
        np.minimum(xs[np.newaxis, :], W - 1 - xs[np.newaxis, :]),
        np.minimum(ys[:, np.newaxis], H - 1 - ys[:, np.newaxis]),
    )
    return np.clip(dist / max(min(W, H) / 2.0, 1.0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------
def stitch(cyl1: np.ndarray, mask1: np.ndarray,
           cyl2: np.ndarray, mask2: np.ndarray,
           du: float, dv: float) -> np.ndarray:
    """
    Place cyl1 and cyl2 on a canvas with center-weighted alpha blending.
    du > 0 → cyl2 is to the right of cyl1.
    """
    H, W, C = cyl1.shape
    du_i, dv_i = int(round(du)), int(round(dv))

    # Canvas offsets for each image
    c1 = max(0, -du_i);  c2 = max(0,  du_i)
    r1 = max(0, -dv_i);  r2 = max(0,  dv_i)
    pW = max(c1 + W, c2 + W)
    pH = max(r1 + H, r2 + H)

    cv = np.zeros((pH, pW, C), dtype=np.float64)
    aw = np.zeros((pH, pW),    dtype=np.float64)

    for img, mask, r_off, c_off in [(cyl1, mask1, r1, c1),
                                     (cyl2, mask2, r2, c2)]:
        a = make_alpha((H, W))
        a[~mask] = 0.0
        a3 = a[:, :, np.newaxis]
        cv[r_off:r_off+H, c_off:c_off+W] += a3 * img
        aw[r_off:r_off+H, c_off:c_off+W] += a

    denom = np.where(aw > 0, aw, 1.0)[:, :, np.newaxis]
    return np.clip(cv / denom, 0.0, 1.0)


# ---------------------------------------------------------------------------
# H-based canvas + blend (least-squares homography in cylindrical space)
# ---------------------------------------------------------------------------
def build_canvas(cyl1: np.ndarray, mask1: np.ndarray,
                 cyl2: np.ndarray, mask2: np.ndarray,
                 H: np.ndarray) -> tuple:
    """
    H maps cyl2 → cyl1 (pass computeH(cyl2_pts, cyl1_pts)).
    Mirrors mosaic.py build_canvas: warpImage(cyl2, H) puts cyl2 in cyl1's frame.
    Canvas origin logic: x_off = max(0, -x_min), y_off = max(0, -y_min).
    """
    h1, w1 = cyl1.shape[:2]
    h2, w2 = cyl2.shape[:2]
    C = cyl1.shape[2]

    corners = np.array([[0,0,1],[w2,0,1],[0,h2,1],[w2,h2,1]],
                       dtype=np.float64).T
    proj = H @ corners;  proj /= proj[2:3]
    x_min, x_max2 = proj[0].min(), proj[0].max()
    y_min, y_max2 = proj[1].min(), proj[1].max()

    x_off = max(0, int(round(-x_min)))
    y_off = max(0, int(round(-y_min)))
    W_c = int(np.ceil(max(float(w1), x_max2) - min(0.0, x_min)))
    H_c = int(np.ceil(max(float(h1), y_max2) - min(0.0, y_min)))

    # cyl1 layer
    cv1 = np.zeros((H_c, W_c, C), dtype=np.float64)
    cv1[y_off:y_off+h1, x_off:x_off+w1] = cyl1
    a1_src = make_alpha((h1, w1));  a1_src[~mask1] = 0.0
    ca1 = np.zeros((H_c, W_c), dtype=np.float64)
    ca1[y_off:y_off+h1, x_off:x_off+w1] = a1_src

    # warp cyl2 into cyl1's frame
    warped2, _ = warpImage(cyl2, H)
    h2w, w2w = warped2.shape[:2]
    r0 = max(0, int(round(y_min + y_off)))
    c0 = max(0, int(round(x_min + x_off)))
    r1e = min(r0 + h2w, H_c);  c1e = min(c0 + w2w, W_c)
    dr, dc = r1e - r0, c1e - c0

    cv2 = np.zeros((H_c, W_c, C), dtype=np.float64)
    cv2[r0:r1e, c0:c1e] = warped2[:dr, :dc]

    # warp cyl2's mask-weighted alpha through H
    a2_src = make_alpha((h2, w2));  a2_src[~mask2] = 0.0
    warped_a2, _ = warpImage(a2_src[:, :, np.newaxis], H)
    ca2 = np.zeros((H_c, W_c), dtype=np.float64)
    ca2[r0:r1e, c0:c1e] = warped_a2[:dr, :dc, 0]

    return cv1, cv2, ca1, ca2, x_off, y_off


def blend(cv1: np.ndarray, cv2: np.ndarray,
          a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    w1 = a1[:, :, np.newaxis];  w2 = a2[:, :, np.newaxis]
    return np.clip((w1 * cv1 + w2 * cv2) / (w1 + w2 + 1e-8), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Interactive correspondence picker (same API as mosaic.py)
# ---------------------------------------------------------------------------
def pick_correspondences(im1: np.ndarray, im2: np.ndarray,
                         n: int = 8) -> tuple:
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    pad = max(h1, h2)
    canvas = np.zeros((pad, w1 + w2, 3), dtype=np.float64)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(canvas)
    ax.axvline(x=w1 - 0.5, color='yellow', lw=2, ls='--', alpha=0.8)
    ax.set_title(f"Click {n} in LEFT (im1), then {n} in RIGHT (im2)", fontsize=11)
    ax.axis('off')
    plt.tight_layout()

    all_pts = np.array(plt.ginput(2 * n, timeout=300), dtype=np.float64)
    plt.close(fig)

    im1_pts = all_pts[:n].copy()
    im2_pts = all_pts[n:].copy()
    im2_pts[:, 0] -= w1
    return im1_pts, im2_pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    hardcoded = "--hardcoded" in sys.argv

    f = DEFAULT_FOCAL
    if "--focal" in sys.argv:
        idx = sys.argv.index("--focal")
        f = float(sys.argv[idx + 1])

    npts = 8
    if "--npts" in sys.argv:
        idx = sys.argv.index("--npts")
        npts = int(sys.argv[idx + 1])

    im1 = mpl_imread("img1.jpg").astype(np.float64) / 255.0
    im2 = mpl_imread("img2.jpg").astype(np.float64) / 255.0
    H_im, W_im = im1.shape[:2]

    if hardcoded:
        im1_pts, im2_pts = IM1_PTS.copy(), IM2_PTS.copy()
        print(f"Using hardcoded correspondences ({len(IM1_PTS)} pairs).")
    else:
        im1_pts, im2_pts = pick_correspondences(im1, im2, n=npts)
        print(f"\nim1_pts =\n{im1_pts}\nim2_pts =\n{im2_pts}")

    n = len(im1_pts)
    print(f"Focal length  f = {f:.0f} px  |  {n} correspondences  "
          f"(image {W_im}×{H_im})")

    print("Projecting im1 → cylinder …")
    cyl1, mask1 = project_to_cylinder(im1, f)
    print("Projecting im2 → cylinder …")
    cyl2, mask2 = project_to_cylinder(im2, f)

    # --- Method A: median translation (original) ---
    du, dv = estimate_offset(im1_pts, im2_pts, f, W_im, H_im)
    print(f"\n[A] Median translation  du={du:+.1f}  dv={dv:+.1f} px")
    pano_A = stitch(cyl1, mask1, cyl2, mask2, du, dv)

    # --- Method B: least-squares H in cylindrical space ---
    cx, cy = W_im / 2.0, H_im / 2.0
    cyl1_pts = to_cylinder_uv(im1_pts, f, cx, cy)
    cyl2_pts = to_cylinder_uv(im2_pts, f, cx, cy)
    H_cyl = computeH(cyl2_pts, cyl1_pts)   # maps cyl2 → cyl1 (n≥4, least-squares)
    print(f"\n[B] Least-squares H in cylindrical space  ({n} pt pairs):\n{H_cyl}")
    cv1, cv2, ca1, ca2, _, _ = build_canvas(cyl1, mask1, cyl2, mask2, H_cyl)
    pano_B = blend(cv1, cv2, ca1, ca2)

    print(f"\nPanorama A (translation): {pano_A.shape[1]}×{pano_A.shape[0]}")
    print(f"Panorama B (H least-sq):  {pano_B.shape[1]}×{pano_B.shape[0]}")

    # ------------------------------------------------------------------ plot
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(f"Cylindrical Panorama  f={f:.0f} px  |  {n} correspondences",
                 fontsize=12, fontweight="bold")
    gs = fig.add_gridspec(3, 2, hspace=0.08, wspace=0.04)

    ax00 = fig.add_subplot(gs[0, 0]); ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0]); ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, :])   # full-width bottom row

    ax00.imshow(im1);              ax00.set_title(f"im1  ({W_im}×{H_im})")
    ax01.imshow(im2);              ax01.set_title(f"im2  ({im2.shape[1]}×{im2.shape[0]})")
    ax10.imshow(np.clip(cyl1,0,1)); ax10.set_title(f"cyl(im1)  f={f:.0f}")
    ax11.imshow(np.clip(cyl2,0,1)); ax11.set_title(f"cyl(im2)  f={f:.0f}")
    ax20.imshow(np.clip(pano_B,0,1))
    ax20.set_title(f"[B] Least-squares H  {pano_B.shape[1]}×{pano_B.shape[0]}")

    for ax in (ax00, ax01, ax10, ax11, ax20):
        ax.axis("off")

    plt.savefig("cylindrical_panorama.png", dpi=150, bbox_inches="tight")
    print("Saved → cylindrical_panorama.png")

    # Save comparison figure separately
    fig2, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig2.suptitle(f"Comparison  f={f:.0f} px  |  {n} pts", fontsize=11,
                  fontweight="bold")
    axes[0].imshow(np.clip(pano_A, 0, 1))
    axes[0].set_title(f"[A] Median translation  {pano_A.shape[1]}×{pano_A.shape[0]}")
    axes[0].axis("off")
    axes[1].imshow(np.clip(pano_B, 0, 1))
    axes[1].set_title(f"[B] Least-squares H  {pano_B.shape[1]}×{pano_B.shape[0]}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("cylindrical_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved → cylindrical_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
