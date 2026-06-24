"""
auto_mosaic.py  —  Part B: Automatic Image Stitching
Reference: Brown et al., "Multi-Image Matching using Multi-Scale Oriented Patches" (MOPS)

Pipeline
--------
1. Harris Interest Point Detector  (harris.py, single scale)
2. Adaptive Non-Maximal Suppression (ANMS)
3. Feature Descriptor extraction + Lowe's ratio-test matching
4. 4-point RANSAC homography
5. Mosaic (same alpha-blending as Part A)

Usage
-----
    uv run python auto_mosaic.py                # auto only; loads saved manual pts if present
    uv run python auto_mosaic.py --debug        # also saves corner / match debug figures
    uv run python auto_mosaic.py --manual 01    # interactively pick lobby0↔lobby1 points
    uv run python auto_mosaic.py --manual 12    # interactively pick lobby1↔lobby2 points
    uv run python auto_mosaic.py --manual all   # pick both pairs in sequence

Manual points are saved to manual_pts_01.npz / manual_pts_12.npz and loaded automatically.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.image import imread as mpl_imread
from skimage.color import rgb2gray
from skimage.transform import resize

from harris import get_harris_corners, dist2
from homography import computeH, warpImage


# ============================================================================
# Step 2 — Adaptive Non-Maximal Suppression (ANMS)
# ============================================================================
def anms(h_map: np.ndarray, coords: np.ndarray,
         n_best: int = 1000, c_robust: float = 0.9,
         pre_filter: int = 3000) -> np.ndarray:
    """
    Select n_best spatially-uniform corners from coords.

    Algorithm (Brown et al. §3.1):
        For each corner i, compute the suppression radius
            r_i = min_{j : f(x_j) > c_robust · f(x_i)}  ||x_i − x_j||
        Keep the n_best corners with the largest r_i.

    coords: (2, N)  [ys; xs] from get_harris_corners
    Returns (2, min(N, n_best)) selected corners.

    pre_filter: cap to the top-pre_filter strongest corners before
    building the N×N distance matrix (avoids O(N²) memory blow-up).
    """
    n = coords.shape[1]
    if n <= n_best:
        return coords

    ys, xs = coords[0], coords[1]
    strength = h_map[ys, xs]

    # ── Pre-filter to the strongest pre_filter corners ──────────────────────
    if n > pre_filter:
        top_idx = np.argpartition(-strength, pre_filter)[:pre_filter]
        coords   = coords[:, top_idx]
        strength = strength[top_idx]
        n        = pre_filter

    if n <= n_best:
        return coords

    # ── Compute pairwise squared Euclidean distances ─────────────────────────
    pts = coords.T.astype(np.float64)   # (n, 2) [y, x]
    D2  = dist2(pts, pts)               # (n, n)

    # stronger[i, j] = True  iff  f(x_j) > c_robust · f(x_i)
    stronger = strength[np.newaxis, :] > c_robust * strength[:, np.newaxis]
    # r_i² = min over stronger neighbours; inf if none exists
    D2_masked = np.where(stronger, D2, np.inf)
    r2 = D2_masked.min(axis=1)

    order = np.argsort(-r2)
    return coords[:, order[:n_best]]


# ============================================================================
# Step 3a — Feature Descriptor Extraction
# ============================================================================
def extract_descriptors(gray_im: np.ndarray, coords: np.ndarray,
                         patch_size: int = 40, desc_size: int = 8,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    For each corner extract a patch_size×patch_size neighbourhood,
    downsample to desc_size×desc_size (implicit Gaussian blur via area
    averaging), then bias/gain normalise (subtract mean, divide by std).

    coords: (2, N) [ys; xs]
    Returns
        descs : (N, desc_size²) float64
        valid : (N,)  bool — False when patch extends outside image border
    """
    half = patch_size // 2
    H, W = gray_im.shape
    N    = coords.shape[1]

    descs = np.zeros((N, desc_size * desc_size), dtype=np.float64)
    valid = np.ones(N, dtype=bool)

    for i in range(N):
        y, x = int(coords[0, i]), int(coords[1, i])
        r0, r1 = y - half, y + half
        c0, c1 = x - half, x + half

        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            valid[i] = False
            continue

        patch = gray_im[r0:r1, c0:c1]
        small = resize(patch, (desc_size, desc_size), anti_aliasing=True)

        mu, sigma = small.mean(), small.std()
        descs[i]  = ((small - mu) / (sigma + 1e-8)).ravel()

    return descs, valid


# ============================================================================
# Step 3b — Feature Matching  (Lowe's ratio test)
# ============================================================================
def match_features(descs1: np.ndarray, descs2: np.ndarray,
                   ratio_thresh: float = 0.6) -> np.ndarray:
    """
    For each descriptor in descs1 find the two nearest neighbours in descs2.
    Accept the match only if

        dist_1st / dist_2nd  <  ratio_thresh

    (Lowe 2004; threshold ≈ 0.6 from MOPS paper Figure 6b.)

    Returns (M, 2) int32 array of (idx_in_descs1, idx_in_descs2) pairs.
    """
    D2 = dist2(descs1, descs2)     # (N1, N2) squared L2 distances

    pairs = []
    for i in range(len(descs1)):
        row = D2[i]
        if len(row) < 2:
            continue

        j1, j2 = np.argpartition(row, 2)[:2]
        if row[j2] < row[j1]:
            j1, j2 = j2, j1

        d1 = np.sqrt(max(row[j1], 0.0))
        d2 = np.sqrt(max(row[j2], 0.0))

        if d2 > 1e-10 and d1 / d2 < ratio_thresh:
            pairs.append((i, j1))

    return (np.array(pairs, dtype=np.int32)
            if pairs else np.empty((0, 2), dtype=np.int32))


# ============================================================================
# Step 4 — 4-point RANSAC Homography
# ============================================================================
def ransac_homography(pts1: np.ndarray, pts2: np.ndarray,
                      n_iter: int = 1000, inlier_thresh: float = 4.0,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust homography estimation via 4-point RANSAC.

    H maps pts2 → pts1  (convention: computeH(pts2, pts1)).

    Each iteration:
        1. Sample 4 random correspondences, compute H.
        2. Forward-map all pts2 through H, count inliers:
               inlier ↔ ||H · p2 − p1|| < inlier_thresh
        3. Keep the H with the most inliers.
    Final H is re-estimated from all inlier pairs.

    Returns
        H_best   : (3, 3)
        inl_mask : (N,)  bool
    """
    N   = len(pts1)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(N, dtype=bool)

    pts2_h = np.hstack([pts2, np.ones((N, 1))])    # (N, 3) homogeneous

    for _ in range(n_iter):
        idx = rng.choice(N, 4, replace=False)
        try:
            H = computeH(pts2[idx], pts1[idx])
        except Exception:
            continue

        mapped = (H @ pts2_h.T).T                  # (N, 3)
        w3     = mapped[:, 2]
        degenerate = np.abs(w3) < 1e-6
        w3_safe    = np.where(degenerate, 1.0, w3)
        xy         = mapped[:, :2] / w3_safe[:, np.newaxis]
        dists      = np.where(degenerate, np.inf,
                              np.linalg.norm(xy - pts1, axis=1))
        mask = dists < inlier_thresh

        if mask.sum() > best_mask.sum():
            best_mask = mask

    # Re-estimate H with all inliers
    if best_mask.sum() >= 4:
        H_final = computeH(pts2[best_mask], pts1[best_mask])
    else:
        H_final = np.eye(3)

    return H_final, best_mask


# ============================================================================
# Full auto-stitch pipeline  (Steps 1-4 + warp)
# ============================================================================
def auto_stitch(im1: np.ndarray, im2: np.ndarray,
                n_corners: int = 1000,
                ratio_thresh: float = 0.6,
                n_ransac: int = 1000,
                inlier_thresh: float = 4.0,
                label: str = "") -> tuple:
    """
    Run the full pipeline on a pair of images.

    H maps im2 → im1 (so warpImage(im2, H) puts im2 in im1's frame).
    Returns (H, pts1, pts2, inlier_mask, mosaic).
    pts are (M, 2) in (x, y) order.
    """
    tag = f"  [{label}]" if label else " "

    gray1 = rgb2gray(im1)
    gray2 = rgb2gray(im2)

    # Step 1 — Harris
    h1_map, coords1 = get_harris_corners(gray1)
    h2_map, coords2 = get_harris_corners(gray2)
    print(f"{tag} Harris:  {coords1.shape[1]} + {coords2.shape[1]} corners")

    # Step 2 — ANMS
    coords1 = anms(h1_map, coords1, n_best=n_corners)
    coords2 = anms(h2_map, coords2, n_best=n_corners)
    print(f"{tag} ANMS:    {coords1.shape[1]} + {coords2.shape[1]} kept")

    # Step 3a — Descriptors
    descs1, v1 = extract_descriptors(gray1, coords1)
    descs2, v2 = extract_descriptors(gray2, coords2)
    coords1, descs1 = coords1[:, v1], descs1[v1]
    coords2, descs2 = coords2[:, v2], descs2[v2]

    # Step 3b — Matching
    matches = match_features(descs1, descs2, ratio_thresh=ratio_thresh)
    print(f"{tag} Lowe:    {len(matches)} matches  (ratio < {ratio_thresh})")

    if len(matches) < 4:
        raise ValueError(
            f"Too few matches ({len(matches)}).  "
            "Try increasing ratio_thresh or n_corners.")

    # coords are [ys; xs]; reorder to (x, y) for computeH
    pts1 = np.column_stack([coords1[1, matches[:, 0]],
                             coords1[0, matches[:, 0]]]).astype(np.float64)
    pts2 = np.column_stack([coords2[1, matches[:, 1]],
                             coords2[0, matches[:, 1]]]).astype(np.float64)

    # Step 4 — RANSAC
    H, inl = ransac_homography(pts1, pts2,
                                n_iter=n_ransac, inlier_thresh=inlier_thresh)
    print(f"{tag} RANSAC:  {inl.sum()}/{len(matches)} inliers")

    # Step 5 — Warp + blend
    mosaic = make_mosaic(im1, im2, H)
    return H, pts1, pts2, inl, mosaic


# ============================================================================
# Mosaic helpers  (self-contained — no import from mosaic.py)
# ============================================================================
def _make_alpha(shape: tuple) -> np.ndarray:
    """Centre-weighted alpha: 1 at image centre, 0 at all four edges."""
    H, W = shape
    ys   = np.arange(H, dtype=np.float64)
    xs   = np.arange(W, dtype=np.float64)
    dist = np.minimum(
        np.minimum(xs[np.newaxis, :], W - 1 - xs[np.newaxis, :]),
        np.minimum(ys[:, np.newaxis], H - 1 - ys[:, np.newaxis]),
    )
    return np.clip(dist / max(min(W, H) / 2.0, 1.0), 0.0, 1.0)


def _projected_extent(H: np.ndarray, h: int, w: int,
                       denom_threshold: float = 0.5) -> tuple:
    """
    Bounding box of a (h, w) image projected through H.

    Samples a 7×7 interior grid and excludes points whose perspective
    denominator is below denom_threshold (magnification > 1/denom_threshold ×).
    This prevents narrow-overlap homographies from producing extreme canvas sizes.
    """
    xs = np.linspace(0, w, 7)
    ys = np.linspace(0, h, 7)
    gx, gy = np.meshgrid(xs, ys)
    pts  = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=0)
    proj = H @ pts
    denom = proj[2]
    valid = denom > denom_threshold
    if not valid.any():
        valid = denom > denom.max() * 0.5
    proj = proj[:, valid] / proj[2:3, valid]
    return proj[0].min(), proj[0].max(), proj[1].min(), proj[1].max()


def _corner_extent(H: np.ndarray, h: int, w: int) -> tuple:
    """Exact bounding box from the 4 image corners (same as warpImage uses internally)."""
    corners = np.array([[0, w, 0, w],
                        [0, 0, h, h],
                        [1, 1, 1, 1]], dtype=np.float64)
    proj = H @ corners
    proj /= proj[2:3]
    return proj[0].min(), proj[0].max(), proj[1].min(), proj[1].max()


def _build_canvas(im1: np.ndarray, im2: np.ndarray,
                  H: np.ndarray) -> tuple:
    """
    Place im1 at its natural position and warp im2 through H into a
    shared canvas.  Returns (canvas_im1, canvas_im2, alpha1, alpha2, x_off, y_off).

    Canvas SIZE is determined by _projected_extent (denom-filtered) to avoid
    extreme blow-up from narrow-overlap homographies.

    Warped-image PLACEMENT uses the exact 4-corner projection so that the
    warpImage output — whose internal coordinate origin equals the 4-corner
    x_min/y_min — is positioned correctly in the canvas.
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    C      = im1.shape[2]

    # Canvas size: use denom-filtered extent to cap blow-up
    x_min_c, x_max_c, y_min_c, y_max_c = _projected_extent(H, h2, w2)
    x_off = max(0, int(round(-x_min_c)))
    y_off = max(0, int(round(-y_min_c)))
    W_c   = int(np.ceil(max(float(w1), x_max_c) - min(0.0, x_min_c)))
    H_c   = int(np.ceil(max(float(h1), y_max_c) - min(0.0, y_min_c)))

    # ── Place im1 ────────────────────────────────────────────────────────────
    canvas1 = np.zeros((H_c, W_c, C), dtype=np.float64)
    canvas1[y_off:y_off + h1, x_off:x_off + w1] = im1
    alpha1 = np.zeros((H_c, W_c), dtype=np.float64)
    alpha1[y_off:y_off + h1, x_off:x_off + w1] = _make_alpha((h1, w1))

    # ── Warp im2 ─────────────────────────────────────────────────────────────
    # warpImage's own origin = exact 4-corner x_min / y_min of im2 in im1 space
    x_min_w, _, y_min_w, _ = _corner_extent(H, h2, w2)

    warped2, _  = warpImage(im2, H)
    h2w, w2w    = warped2.shape[:2]

    # Canvas position of warped2[0, 0] = im1-space (x_min_w, y_min_w)
    r_start = int(round(y_min_w + y_off))
    c_start = int(round(x_min_w + x_off))

    # Which region of the canvas receives content from warped2?
    r0_cv = max(0, r_start);          c0_cv = max(0, c_start)
    r1_cv = min(r_start + h2w, H_c);  c1_cv = min(c_start + w2w, W_c)

    if r0_cv < r1_cv and c0_cv < c1_cv:
        # Corresponding rows / cols inside warped2
        r0_w = r0_cv - r_start;  r1_w = r1_cv - r_start
        c0_w = c0_cv - c_start;  c1_w = c1_cv - c_start

        canvas2 = np.zeros((H_c, W_c, C), dtype=np.float64)
        canvas2[r0_cv:r1_cv, c0_cv:c1_cv] = warped2[r0_w:r1_w, c0_w:c1_w]

        alpha2_src   = _make_alpha((h2, w2))[:, :, np.newaxis]
        warped_a2, _ = warpImage(alpha2_src, H)
        alpha2 = np.zeros((H_c, W_c), dtype=np.float64)
        alpha2[r0_cv:r1_cv, c0_cv:c1_cv] = warped_a2[r0_w:r1_w, c0_w:c1_w, 0]
    else:
        canvas2 = np.zeros((H_c, W_c, C), dtype=np.float64)
        alpha2  = np.zeros((H_c, W_c),    dtype=np.float64)

    return canvas1, canvas2, alpha1, alpha2, x_off, y_off


def _blend(c1: np.ndarray, c2: np.ndarray,
           a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Weighted average: (a1·im1 + a2·im2) / (a1 + a2 + ε), clipped [0, 1]."""
    a1 = a1[:, :, np.newaxis]
    a2 = a2[:, :, np.newaxis]
    return np.clip((a1 * c1 + a2 * c2) / (a1 + a2 + 1e-8), 0.0, 1.0)


def make_mosaic(im1: np.ndarray, im2: np.ndarray,
                H: np.ndarray) -> np.ndarray:
    c1, c2, a1, a2, _, _ = _build_canvas(im1, im2, H)
    return _blend(c1, c2, a1, a2)


def crop_to_content(mosaic: np.ndarray,
                    threshold: float = 0.01) -> np.ndarray:
    """Remove rows / columns whose max pixel value is below threshold."""
    gray     = mosaic.mean(axis=2)
    row_mask = gray.max(axis=1) > threshold
    col_mask = gray.max(axis=0) > threshold
    rows     = np.where(row_mask)[0]
    cols     = np.where(col_mask)[0]
    if rows.size == 0 or cols.size == 0:
        return mosaic
    return mosaic[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


# ============================================================================
# Three-image mosaic  (reference frame = im1)
# ============================================================================
def make_mosaic_3(im1: np.ndarray, im2: np.ndarray, im3: np.ndarray,
                  H12: np.ndarray, H13: np.ndarray) -> np.ndarray:
    """
    Build a three-image panorama in im1's coordinate frame.

    H12 maps im2 → im1 space.
    H13 maps im3 → im1 space.

    All three images are alpha-blended in a common canvas.
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    h3, w3 = im3.shape[:2]
    C      = im1.shape[2]

    # ── Canvas extent ─────────────────────────────────────────────────────────
    x1_min, x1_max = 0.0, float(w1)
    y1_min, y1_max = 0.0, float(h1)
    x2_min, x2_max, y2_min, y2_max = _projected_extent(H12, h2, w2)
    x3_min, x3_max, y3_min, y3_max = _projected_extent(H13, h3, w3)

    x_min = min(x1_min, x2_min, x3_min)
    x_max = max(x1_max, x2_max, x3_max)
    y_min = min(y1_min, y2_min, y3_min)
    y_max = max(y1_max, y2_max, y3_max)

    x_off = max(0, int(round(-x_min)))
    y_off = max(0, int(round(-y_min)))
    W_c   = int(np.ceil(x_max - x_min))
    H_c   = int(np.ceil(y_max - y_min))

    # ── Place im1 ────────────────────────────────────────────────────────────
    canvas1 = np.zeros((H_c, W_c, C), dtype=np.float64)
    canvas1[y_off:y_off + h1, x_off:x_off + w1] = im1
    alpha1  = np.zeros((H_c, W_c), dtype=np.float64)
    alpha1[y_off:y_off + h1, x_off:x_off + w1] = _make_alpha((h1, w1))

    def _warp_and_place(im: np.ndarray, H: np.ndarray):
        h_src, w_src = im.shape[:2]
        # warpImage's own origin = exact 4-corner min
        xm_w, _, ym_w, _ = _corner_extent(H, h_src, w_src)

        warped, _ = warpImage(im, H)
        h_w, w_w  = warped.shape[:2]

        r_start = int(round(ym_w + y_off))
        c_start = int(round(xm_w + x_off))

        r0_cv = max(0, r_start);         c0_cv = max(0, c_start)
        r1_cv = min(r_start + h_w, H_c); c1_cv = min(c_start + w_w, W_c)

        out_im = np.zeros((H_c, W_c, C), dtype=np.float64)
        out_a  = np.zeros((H_c, W_c),    dtype=np.float64)

        if r0_cv < r1_cv and c0_cv < c1_cv:
            r0_w = r0_cv - r_start; r1_w = r1_cv - r_start
            c0_w = c0_cv - c_start; c1_w = c1_cv - c_start
            out_im[r0_cv:r1_cv, c0_cv:c1_cv] = warped[r0_w:r1_w, c0_w:c1_w]

            a_src        = _make_alpha(im.shape[:2])[:, :, np.newaxis]
            warped_a, _  = warpImage(a_src, H)
            out_a[r0_cv:r1_cv, c0_cv:c1_cv] = warped_a[r0_w:r1_w, c0_w:c1_w, 0]

        return out_im, out_a

    canvas2, alpha2 = _warp_and_place(im2, H12)
    canvas3, alpha3 = _warp_and_place(im3, H13)

    a1 = alpha1[:, :, np.newaxis]
    a2 = alpha2[:, :, np.newaxis]
    a3 = alpha3[:, :, np.newaxis]
    return np.clip(
        (a1 * canvas1 + a2 * canvas2 + a3 * canvas3) / (a1 + a2 + a3 + 1e-8),
        0.0, 1.0,
    )


# ============================================================================
# Visualization helpers
# ============================================================================
def plot_corners(im: np.ndarray, coords_all: np.ndarray,
                 coords_anms: np.ndarray, title: str = "",
                 save: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for ax, c, lbl in zip(
        axes,
        [coords_all, coords_anms],
        [f"All Harris  ({coords_all.shape[1]})",
         f"After ANMS  ({coords_anms.shape[1]})"],
    ):
        ax.imshow(im)
        ax.scatter(c[1], c[0], s=4, c="red", linewidths=0)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        print(f"    → {save}")
    plt.show()


def _save_corner_plots(named_images: dict) -> None:
    """
    For each image in named_images, run Harris + ANMS and save a side-by-side
    figure: left = all Harris corners, right = after ANMS (n_best=1000).

    Output files: corners_{name}.png
    """
    for name, im in named_images.items():
        gray     = rgb2gray(im)
        h_map, coords_all = get_harris_corners(gray)
        coords_anms       = anms(h_map, coords_all, n_best=1000)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"{name}  —  Harris corners  →  ANMS",
                     fontsize=13, fontweight="bold")

        for ax, coords, title in zip(
            axes,
            [coords_all, coords_anms],
            [f"Harris  ({coords_all.shape[1]} corners)",
             f"ANMS  (top {coords_anms.shape[1]}, spatially uniform)"],
        ):
            ax.imshow(im)
            ax.scatter(coords[1], coords[0],
                       s=3, c="red", linewidths=0, alpha=0.6)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        fname = f"corners_{name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}  "
              f"({coords_all.shape[1]} → {coords_anms.shape[1]} corners)")


def plot_lowe_matches(im1: np.ndarray, im2: np.ndarray,
                      pts1: np.ndarray, pts2: np.ndarray,
                      title: str = "", save: str | None = None) -> None:
    """
    Show ALL matches after Lowe's ratio test (before RANSAC) in yellow.
    pts1, pts2: (M, 2) all accepted Lowe pairs.
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float64)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(canvas)

    p2s = pts2.copy()
    p2s[:, 0] += w1

    for (x1, y1), (x2, y2) in zip(pts1, p2s):
        ax.plot([x1, x2], [y1, y2], color="yellow", lw=0.6, alpha=0.5)
    ax.scatter(pts1[:, 0], pts1[:, 1], s=8, c="yellow", linewidths=0)
    ax.scatter(p2s[:, 0],  p2s[:, 1],  s=8, c="yellow", linewidths=0)

    ax.set_title(f"{title}  —  {len(pts1)} Lowe matches (before RANSAC)", fontsize=11)
    ax.axvline(x=w1, color="cyan", lw=1.5, ls="--", alpha=0.8)
    ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        print(f"  Saved {save}")
    plt.show()


def plot_matches(im1: np.ndarray, im2: np.ndarray,
                 pts1: np.ndarray, pts2: np.ndarray,
                 inl: np.ndarray, title: str = "",
                 save: str | None = None) -> None:
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.float64)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(canvas)

    p2s = pts2.copy()
    p2s[:, 0] += w1

    for mask, color in [(~inl, "red"), (inl, "lime")]:
        p1m, p2m = pts1[mask], p2s[mask]
        for (x1, y1), (x2, y2) in zip(p1m, p2m):
            ax.plot([x1, x2], [y1, y2], color=color, lw=0.7, alpha=0.6)
        ax.scatter(p1m[:, 0], p1m[:, 1], s=8, c=color, linewidths=0)
        ax.scatter(p2m[:, 0], p2m[:, 1], s=8, c=color, linewidths=0)

    ax.set_title(f"{title}  —  {inl.sum()}/{len(inl)} inliers (green)", fontsize=11)
    ax.axvline(x=w1, color="yellow", lw=1.5, ls="--", alpha=0.8)
    ax.axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        print(f"    → {save}")
    plt.show()


# ============================================================================
# Manual correspondences for Part A vs Part B comparison
# ============================================================================
# These are auto-loaded from manual_pts_01.npz / manual_pts_12.npz if present.
# You can also hard-code them here as (N, 2) float64 arrays.
MANUAL_PTS: dict[str, tuple] = {
    "01": (None, None),
    "12": (None, None),
}

_MANUAL_FILES = {"01": "manual_pts_01.npz", "12": "manual_pts_12.npz"}


def _load_manual_pts() -> None:
    """Load any saved manual correspondences into MANUAL_PTS."""
    import os
    for key, fname in _MANUAL_FILES.items():
        if os.path.exists(fname) and MANUAL_PTS[key] == (None, None):
            data = np.load(fname)
            MANUAL_PTS[key] = (data["pts_a"], data["pts_b"])
            print(f"  Loaded manual pts '{key}' from {fname}  "
                  f"({len(data['pts_a'])} pairs)")


def _save_manual_pts(key: str, pts_a: np.ndarray, pts_b: np.ndarray) -> None:
    fname = _MANUAL_FILES[key]
    np.savez(fname, pts_a=pts_a, pts_b=pts_b)
    print(f"  Saved {len(pts_a)} manual pairs → {fname}")


def pick_correspondences(im_a: np.ndarray, im_b: np.ndarray,
                         n: int = 8,
                         label_a: str = "Image A",
                         label_b: str = "Image B") -> tuple:
    """
    Show im_a and im_b side-by-side and collect 2n clicks via ginput:
    first n in im_a (left), then n in im_b (right).

    Returns (pts_a, pts_b) as (n, 2) float64 arrays in (x, y) order.
    """
    h_a, w_a = im_a.shape[:2]
    h_b, w_b = im_b.shape[:2]
    H_pad    = max(h_a, h_b)

    canvas = np.zeros((H_pad, w_a + w_b, 3), dtype=np.float64)
    canvas[:h_a, :w_a]      = im_a
    canvas[:h_b, w_a:w_a + w_b] = im_b

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(canvas)
    ax.axvline(x=w_a - 0.5, color="yellow", lw=2, ls="--", alpha=0.8)
    ax.set_title(
        f"Click {n} points in the LEFT image ({label_a}),\n"
        f"then {n} corresponding points in the RIGHT image ({label_b}).\n"
        "Close window when done (or wait for timeout).",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()

    all_pts = np.array(plt.ginput(2 * n, timeout=0), dtype=np.float64)
    plt.close(fig)

    if len(all_pts) < 2 * n:
        raise RuntimeError(
            f"Expected {2*n} clicks, got {len(all_pts)}. "
            "Re-run --manual to pick again.")

    pts_a = all_pts[:n].copy()
    pts_b = all_pts[n:].copy()
    pts_b[:, 0] -= w_a          # remove the x-offset from the side-by-side display
    return pts_a, pts_b


def run_manual_picking(pairs: list[str],
                       im0: np.ndarray, im1: np.ndarray, im2: np.ndarray,
                       n: int = 8) -> None:
    """
    Interactively pick manual correspondences for the requested pairs and save them.
    pairs: list of keys from {"01", "12"}.
    """
    images = {"0": im0, "1": im1, "2": im2}
    names  = {"0": "lobby0", "1": "lobby1", "2": "lobby2"}

    for key in pairs:
        a, b = key[0], key[1]
        print(f"\n── Manual picking: {names[a]} ↔ {names[b]}  ({n} pairs) ───")
        print("  Click in the LEFT panel first, then the RIGHT panel.")
        pts_a, pts_b = pick_correspondences(
            images[a], images[b], n=n,
            label_a=names[a], label_b=names[b])
        MANUAL_PTS[key] = (pts_a, pts_b)
        _save_manual_pts(key, pts_a, pts_b)
        print(f"  Picked {len(pts_a)} pairs.")
        print(f"  pts_{names[a]} =\n{pts_a}")
        print(f"  pts_{names[b]} =\n{pts_b}")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    debug  = "--debug"  in sys.argv
    manual = "--manual" in sys.argv

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading images …")
    im0 = mpl_imread("lobby0.jpg").astype(np.float64) / 255.0
    im1 = mpl_imread("lobby1.jpg").astype(np.float64) / 255.0
    im2 = mpl_imread("lobby2.jpg").astype(np.float64) / 255.0
    print(f"  lobby0: {im0.shape[1]}×{im0.shape[0]}")
    print(f"  lobby1: {im1.shape[1]}×{im1.shape[0]}")
    print(f"  lobby2: {im2.shape[1]}×{im2.shape[0]}")

    # ── Manual picking mode ───────────────────────────────────────────────────
    if manual:
        idx = sys.argv.index("--manual")
        arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "all"
        pairs = ["01", "12"] if arg == "all" else [arg]
        n_pts = int(sys.argv[sys.argv.index("--npts") + 1]) \
                if "--npts" in sys.argv else 8
        run_manual_picking(pairs, im0, im1, im2, n=n_pts)
        print("\nManual picking done. Re-run without --manual to generate mosaics.")
        return

    # ── Auto-load saved manual pts ────────────────────────────────────────────
    _load_manual_pts()

    # ── Harris + ANMS visualizations ─────────────────────────────────────────
    print("\n── Harris / ANMS corner images ─────────────────────────────")
    _save_corner_plots({"lobby0": im0, "lobby1": im1, "lobby2": im2})

    # ════════════════ Set 1 : lobby0 + lobby1 ════════════════════════════════
    print("\n── Set 1: lobby0 + lobby1 ──────────────────────────────────")
    H01, p0_01, p1_01, inl01, mos01_raw = auto_stitch(im0, im1, label="S1")
    mos01 = crop_to_content(mos01_raw)
    print(f"  mosaic: {mos01.shape[1]}×{mos01.shape[0]}")

    plot_lowe_matches(im0, im1, p0_01, p1_01,
                      "Set 1: lobby0 ↔ lobby1",
                      save="lowe_matches_01.png")
    plot_matches(im0, im1, p0_01, p1_01, inl01,
                 "Set 1: lobby0 ↔ lobby1",
                 save="ransac_matches_01.png")

    if debug:
        gray0 = rgb2gray(im0)
        h0m, c0_all = get_harris_corners(gray0)
        c0_anms     = anms(h0m, c0_all)
        plot_corners(im0, c0_all, c0_anms, "Set 1 – lobby0 corners",
                     save="debug_corners_0.png")

    p0_man, p1_man = MANUAL_PTS["01"]
    mos01_man = (crop_to_content(make_mosaic(im0, im1, computeH(p1_man, p0_man)))
                 if p0_man is not None else None)

    # ════════════════ Set 2 : lobby1 + lobby2 ════════════════════════════════
    print("\n── Set 2: lobby1 + lobby2 ──────────────────────────────────")
    H12, p1_12, p2_12, inl12, mos12_raw = auto_stitch(im1, im2, label="S2")
    mos12 = crop_to_content(mos12_raw)
    print(f"  mosaic: {mos12.shape[1]}×{mos12.shape[0]}")

    plot_lowe_matches(im1, im2, p1_12, p2_12,
                      "Set 2: lobby1 ↔ lobby2",
                      save="lowe_matches_12.png")
    plot_matches(im1, im2, p1_12, p2_12, inl12,
                 "Set 2: lobby1 ↔ lobby2",
                 save="ransac_matches_12.png")

    if debug:
        pass

    p1_man2, p2_man2 = MANUAL_PTS["12"]
    if p1_man2 is not None:
        H_man12 = computeH(p2_man2, p1_man2)   # maps lobby2 → lobby1
        mos12_man = crop_to_content(make_mosaic(im1, im2, H_man12))
    else:
        H_man12   = None
        mos12_man = None

    # ════════════════ Set 3 : lobby0 + lobby1 + lobby2 (pivot = lobby1) ══════
    #   Auto:   inv(H01) maps lobby0 → lobby1,  H12      maps lobby2 → lobby1
    #   Manual: inv(H_man01) maps lobby0 → lobby1, H_man12 maps lobby2 → lobby1
    print("\n── Set 3: lobby0 + lobby1 + lobby2  (pivot = lobby1) ───────")
    H0_to_1 = np.linalg.inv(H01)
    mos_full = crop_to_content(make_mosaic_3(im1, im0, im2, H0_to_1, H12))
    print(f"  auto mosaic: {mos_full.shape[1]}×{mos_full.shape[0]}")

    if p0_man is not None and H_man12 is not None:
        H_man01    = computeH(p1_man, p0_man)   # maps lobby1 → lobby0
        H_man0to1  = np.linalg.inv(H_man01)     # maps lobby0 → lobby1
        mos_full_man = crop_to_content(
            make_mosaic_3(im1, im0, im2, H_man0to1, H_man12))
        print(f"  manual mosaic: {mos_full_man.shape[1]}×{mos_full_man.shape[0]}")
    else:
        mos_full_man = None
        print("  (manual pts for both pairs needed for manual Set 3)")

    # ── Save individual results ───────────────────────────────────────────────
    plt.imsave("auto_result_01.png",  np.clip(mos01,    0, 1))
    plt.imsave("auto_result_12.png",  np.clip(mos12,    0, 1))
    plt.imsave("auto_result_012.png", np.clip(mos_full, 0, 1))
    saved = ["auto_result_01.png", "auto_result_12.png", "auto_result_012.png"]
    if mos01_man is not None:
        plt.imsave("manual_result_01.png",  np.clip(mos01_man,  0, 1)); saved.append("manual_result_01.png")
    if mos12_man is not None:
        plt.imsave("manual_result_12.png",  np.clip(mos12_man,  0, 1)); saved.append("manual_result_12.png")
    if mos_full_man is not None:
        plt.imsave("manual_result_012.png", np.clip(mos_full_man, 0, 1)); saved.append("manual_result_012.png")
    print("\nSaved: " + ", ".join(saved))

    # ── Comparison figure (manual Part A vs auto Part B) ─────────────────────
    print("\n── Summary figure ──────────────────────────────────────────")
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.suptitle("Part A  (manual)  vs  Part B  (auto) — Lobby Panoramas",
                 fontsize=14, fontweight="bold")

    row0_data = [
        (mos01_man,   "Set 1 Manual\nlobby0 + lobby1"),
        (mos12_man,   "Set 2 Manual\nlobby1 + lobby2"),
        (mos_full_man,"Set 3 Manual\nlobby0 + lobby1 + lobby2"),
    ]
    for ax, (m, t) in zip(axes[0], row0_data):
        if m is not None:
            ax.imshow(np.clip(m, 0, 1))
        else:
            ax.set_facecolor("#222")
            ax.text(0.5, 0.5,
                    "Manual pts not provided\n(fill MANUAL_PTS to compare)",
                    ha="center", va="center", color="white", fontsize=10,
                    transform=ax.transAxes)
        ax.set_title(t, fontsize=10)
        ax.axis("off")

    row1_data = [
        (mos01,   f"Set 1 Auto\nlobby0 + lobby1  ({inl01.sum()} inliers)"),
        (mos12,   f"Set 2 Auto\nlobby1 + lobby2  ({inl12.sum()} inliers)"),
        (mos_full, "Set 3 Auto\nlobby0 + lobby1 + lobby2"),
    ]
    for ax, (m, t) in zip(axes[1], row1_data):
        ax.imshow(np.clip(m, 0, 1))
        ax.set_title(t, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("auto_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: auto_summary.png")
    plt.show()


if __name__ == "__main__":
    main()
