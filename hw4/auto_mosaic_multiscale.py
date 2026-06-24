"""
auto_mosaic_multiscale.py — Multi-scale Harris + Feature Descriptor

Extension of auto_mosaic.py: instead of detecting Harris corners on the
original image only, we build a scale pyramid and run Harris at each level.

Scale pyramid
-------------
SCALES = [1.0, 0.5, 0.25]
  Scale 1.0  → original image          → 40×40 patch ≈  40 px receptive field
  Scale 0.5  → 2× downsampled image    → 40×40 patch ≈  80 px receptive field
  Scale 0.25 → 4× downsampled image    → 40×40 patch ≈ 160 px receptive field

Key differences from single-scale
-----------------------------------
1. harris_multiscale()  — runs harris.py at each scale, reprojects back to
   original coordinates.
2. anms_multiscale()    — pools corners from all scales, normalises h-map
   strengths per scale before running suppression.
3. extract_descs_ms()   — each corner's patch is extracted from the scale it
   was detected at (larger receptive field for coarser scales).
4. The rest of the pipeline (matching, RANSAC, blending) is unchanged.

Outputs per run
---------------
  ms_corners_{name}.png    : one panel per scale + combined overlay
  ms_anms_{name}.png       : ANMS result coloured by source scale
  ms_lowe_{pair}.png       : Lowe matches before RANSAC
  ms_ransac_{pair}.png     : RANSAC inliers (green) / outliers (red)
  ms_result_{pair}.png     : multi-scale mosaic
  ms_comparison.png        : 2-row figure, single-scale (top) vs multi-scale (bottom)

Usage
-----
    uv run python auto_mosaic_multiscale.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.image import imread as mpl_imread
from skimage.color import rgb2gray
from skimage.transform import resize, rescale as sk_rescale

from harris import get_harris_corners, dist2
from homography import computeH, warpImage

# Import unchanged building-blocks from the single-scale version
from auto_mosaic import (
    match_features, ransac_homography,
    make_mosaic, make_mosaic_3, crop_to_content,
    _load_manual_pts, MANUAL_PTS, _MANUAL_FILES,
    _build_canvas, _blend, _make_alpha,
    _corner_extent, _projected_extent,
    plot_lowe_matches, plot_matches,
    auto_stitch,           # single-scale version — used for comparison
)

# ── Scale pyramid ────────────────────────────────────────────────────────────
SCALES = [1.0, 0.5, 0.25]
SCALE_COLORS = ["red", "deepskyblue", "lime"]   # one colour per level


# ============================================================================
# Step 1 — Harris at a single scale, projected back to original coordinates
# ============================================================================
def _harris_at_scale(gray_im: np.ndarray, scale: float,
                     edge_discard: int = 20) -> tuple:
    """
    Run Harris on the image resampled to `scale` (0 < scale ≤ 1).
    Returns (h_map_orig, coords_orig):
        h_map_orig  : (H, W)  Harris response resized to original resolution
        coords_orig : (2, N)  [ys; xs] in original-image pixel coordinates
    """
    if scale == 1.0:
        im_s = gray_im
    else:
        im_s = sk_rescale(gray_im, scale, anti_aliasing=True)

    h_map_s, coords_s = get_harris_corners(im_s, edge_discard=edge_discard)

    # Reproject corners to original space
    coords_orig = np.round(coords_s.astype(float) / scale).astype(int)

    # Clip to valid region in original image
    H, W = gray_im.shape
    margin = int(np.ceil(edge_discard / scale))
    valid = (
        (coords_orig[0] > margin) & (coords_orig[0] < H - margin) &
        (coords_orig[1] > margin) & (coords_orig[1] < W - margin)
    )
    coords_orig = coords_orig[:, valid]

    # Resize h_map to original resolution (for cross-scale ANMS)
    h_map_orig = resize(h_map_s, gray_im.shape, anti_aliasing=True)
    return h_map_orig, coords_orig


# ============================================================================
# Step 2 — Multi-scale ANMS
# ============================================================================
def anms_multiscale(corners_by_scale: list, hmaps_by_scale: list,
                    n_best: int = 1000, c_robust: float = 0.9,
                    pre_filter: int = 3000) -> tuple:
    """
    Pool corners from all scales, normalise h-map strengths per scale to
    [0, 1], then run standard ANMS on the combined set.

    Returns
        selected_corners : (2, n_best)  in original-image coordinates
        scale_labels     : (n_best,)    index into SCALES for each corner
    """
    # Pool corners and record their source scale
    all_corners = np.hstack(corners_by_scale)                      # (2, N_total)
    all_labels  = np.concatenate([
        np.full(c.shape[1], i, dtype=np.int32)
        for i, c in enumerate(corners_by_scale)
    ])

    # Normalise Harris response per scale, then read strength at each corner
    strengths_parts = []
    for hm, cs in zip(hmaps_by_scale, corners_by_scale):
        hm_n = hm / (hm.max() + 1e-10)
        strengths_parts.append(hm_n[cs[0], cs[1]])
    all_strengths = np.concatenate(strengths_parts)

    n = all_corners.shape[1]

    # Pre-filter to top pre_filter by strength
    if n > pre_filter:
        top = np.argpartition(-all_strengths, pre_filter)[:pre_filter]
        all_corners   = all_corners[:, top]
        all_strengths = all_strengths[top]
        all_labels    = all_labels[top]
        n = pre_filter

    if n <= n_best:
        return all_corners, all_labels

    pts  = all_corners.T.astype(np.float64)   # (n, 2)
    D2   = dist2(pts, pts)
    stronger = all_strengths[np.newaxis, :] > c_robust * all_strengths[:, np.newaxis]
    D2_masked = np.where(stronger, D2, np.inf)
    r2   = D2_masked.min(axis=1)

    order = np.argsort(-r2)[:n_best]
    return all_corners[:, order], all_labels[order]


# ============================================================================
# Step 3a — Multi-scale Feature Descriptors
# ============================================================================
def extract_descs_ms(gray_im: np.ndarray,
                     corners: np.ndarray,
                     scale_labels: np.ndarray,
                     scales: list = SCALES,
                     patch_size: int = 40,
                     desc_size: int = 8) -> tuple:
    """
    For each corner, extract the 40×40 patch from the scale image at which
    that corner was detected, then downsample and normalise.

    A corner detected at scale s has a receptive field of
    (patch_size / s) pixels in the original image.

    Returns
        descs : (N, desc_size²) float64
        valid : (N,) bool
    """
    # Cache scaled images
    scaled_imgs = []
    for s in scales:
        scaled_imgs.append(gray_im if s == 1.0
                           else sk_rescale(gray_im, s, anti_aliasing=True))

    half  = patch_size // 2
    N     = corners.shape[1]
    descs = np.zeros((N, desc_size * desc_size), dtype=np.float64)
    valid = np.ones(N, dtype=bool)

    for i in range(N):
        y_orig, x_orig = int(corners[0, i]), int(corners[1, i])
        s    = scales[scale_labels[i]]
        im_s = scaled_imgs[scale_labels[i]]

        y_s  = int(round(y_orig * s))
        x_s  = int(round(x_orig * s))
        H_s, W_s = im_s.shape

        r0, r1 = y_s - half, y_s + half
        c0, c1 = x_s - half, x_s + half
        if r0 < 0 or r1 > H_s or c0 < 0 or c1 > W_s:
            valid[i] = False
            continue

        patch  = im_s[r0:r1, c0:c1]
        small  = resize(patch, (desc_size, desc_size), anti_aliasing=True)
        mu, sg = small.mean(), small.std()
        descs[i] = ((small - mu) / (sg + 1e-8)).ravel()

    return descs, valid


# ============================================================================
# Full multi-scale auto-stitch pipeline
# ============================================================================
def auto_stitch_ms(im1: np.ndarray, im2: np.ndarray,
                   scales: list = SCALES,
                   n_corners: int = 1000,
                   ratio_thresh: float = 0.6,
                   n_ransac: int = 1000,
                   inlier_thresh: float = 4.0,
                   label: str = "") -> tuple:
    """
    Multi-scale pipeline.  Returns
        H, pts1, pts2, inl, mosaic,
        corners_by_scale_1, corners_by_scale_2,
        anms_corners_1, anms_labels_1,
        anms_corners_2, anms_labels_2
    """
    tag = f"  [{label}]" if label else " "
    g1, g2 = rgb2gray(im1), rgb2gray(im2)

    # ── Step 1: Harris at each scale ──────────────────────────────────────────
    hmaps1, cbys1 = [], []
    hmaps2, cbys2 = [], []
    for s in scales:
        hm1, c1 = _harris_at_scale(g1, s)
        hm2, c2 = _harris_at_scale(g2, s)
        hmaps1.append(hm1); cbys1.append(c1)
        hmaps2.append(hm2); cbys2.append(c2)
        print(f"{tag} Scale {s:.2f}:  {c1.shape[1]} + {c2.shape[1]} corners")

    total1 = sum(c.shape[1] for c in cbys1)
    total2 = sum(c.shape[1] for c in cbys2)
    print(f"{tag} Total:    {total1} + {total2} corners across {len(scales)} scales")

    # ── Step 2: Multi-scale ANMS ──────────────────────────────────────────────
    ac1, lbl1 = anms_multiscale(cbys1, hmaps1, n_best=n_corners)
    ac2, lbl2 = anms_multiscale(cbys2, hmaps2, n_best=n_corners)
    print(f"{tag} ANMS:     {ac1.shape[1]} + {ac2.shape[1]} kept")

    # ── Step 3a: Multi-scale descriptors ──────────────────────────────────────
    d1, v1 = extract_descs_ms(g1, ac1, lbl1, scales)
    d2, v2 = extract_descs_ms(g2, ac2, lbl2, scales)
    ac1, d1, lbl1 = ac1[:, v1], d1[v1], lbl1[v1]
    ac2, d2, lbl2 = ac2[:, v2], d2[v2], lbl2[v2]

    # ── Step 3b: Lowe ratio matching ──────────────────────────────────────────
    matches = match_features(d1, d2, ratio_thresh=ratio_thresh)
    print(f"{tag} Lowe:     {len(matches)} matches  (ratio < {ratio_thresh})")

    if len(matches) < 4:
        raise ValueError(f"Too few matches ({len(matches)}). "
                         "Try increasing ratio_thresh or n_corners.")

    pts1 = np.column_stack([ac1[1, matches[:, 0]],
                             ac1[0, matches[:, 0]]]).astype(np.float64)
    pts2 = np.column_stack([ac2[1, matches[:, 1]],
                             ac2[0, matches[:, 1]]]).astype(np.float64)

    # ── Step 4: RANSAC ────────────────────────────────────────────────────────
    H, inl = ransac_homography(pts1, pts2,
                                n_iter=n_ransac, inlier_thresh=inlier_thresh)
    print(f"{tag} RANSAC:   {inl.sum()}/{len(matches)} inliers")

    mosaic = make_mosaic(im1, im2, H)
    return (H, pts1, pts2, inl, mosaic,
            cbys1, cbys2, ac1, lbl1, ac2, lbl2)


# ============================================================================
# Visualizations
# ============================================================================
def plot_ms_corners(im: np.ndarray,
                    corners_by_scale: list,
                    anms_corners: np.ndarray,
                    anms_labels: np.ndarray,
                    scales: list = SCALES,
                    name: str = "",
                    save: str | None = None) -> None:
    """
    Two rows:
      Row 0: one panel per scale (raw Harris output at that scale)
      Row 1: combined raw overlay  |  after ANMS coloured by scale
    """
    n_scales = len(scales)
    fig, axes = plt.subplots(2, n_scales + 1,
                              figsize=(5 * (n_scales + 1), 8))
    fig.suptitle(f"{name}  —  Multi-scale Harris corners",
                 fontsize=13, fontweight="bold")

    # ── Row 0: raw Harris per scale ───────────────────────────────────────────
    for ax, c, s, col in zip(axes[0], corners_by_scale, scales, SCALE_COLORS):
        ax.imshow(im)
        ax.scatter(c[1], c[0], s=3, c=col, linewidths=0, alpha=0.7)
        ax.set_title(f"Scale {s}  ({c.shape[1]} corners)\n"
                     f"receptive field ≈ {int(40/s)}px", fontsize=9)
        ax.axis("off")

    # Last panel of row 0: all scales combined
    ax_all = axes[0, n_scales]
    ax_all.imshow(im)
    for c, col, s in zip(corners_by_scale, SCALE_COLORS, scales):
        ax_all.scatter(c[1], c[0], s=2, c=col, linewidths=0,
                       alpha=0.5, label=f"scale {s}")
    ax_all.legend(markerscale=3, fontsize=8, loc="upper right")
    total = sum(c.shape[1] for c in corners_by_scale)
    ax_all.set_title(f"All scales  ({total} total)", fontsize=9)
    ax_all.axis("off")

    # ── Row 1 left panels: blank (placeholder for symmetry) ───────────────────
    for ax in axes[1, :n_scales]:
        ax.axis("off")

    # ── Row 1 right panel: ANMS result coloured by scale ─────────────────────
    ax_anms = axes[1, n_scales]
    ax_anms.imshow(im)
    for i, (col, s) in enumerate(zip(SCALE_COLORS, scales)):
        mask = anms_labels == i
        if mask.any():
            ax_anms.scatter(anms_corners[1, mask], anms_corners[0, mask],
                            s=6, c=col, linewidths=0, alpha=0.8,
                            label=f"scale {s}  ({mask.sum()})")
    ax_anms.legend(markerscale=2, fontsize=8, loc="upper right")
    ax_anms.set_title(f"After ANMS  ({anms_corners.shape[1]} kept)", fontsize=9)
    ax_anms.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved {save}")
    plt.show()


def plot_comparison(single_mosaic: np.ndarray, multi_mosaic: np.ndarray,
                    label: str = "", save: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f"{label}  —  Single-scale vs Multi-scale",
                 fontsize=13, fontweight="bold")
    axes[0].imshow(np.clip(single_mosaic, 0, 1))
    axes[0].set_title("Single-scale  (scale 1.0 only)", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(np.clip(multi_mosaic, 0, 1))
    axes[1].set_title(f"Multi-scale  ({', '.join(str(s) for s in SCALES)})",
                      fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Saved {save}")
    plt.show()


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    print("Loading images …")
    im0 = mpl_imread("lobby0.jpg").astype(np.float64) / 255.0
    im1 = mpl_imread("lobby1.jpg").astype(np.float64) / 255.0
    im2 = mpl_imread("lobby2.jpg").astype(np.float64) / 255.0
    print(f"  lobby0: {im0.shape[1]}×{im0.shape[0]}")
    print(f"  lobby1: {im1.shape[1]}×{im1.shape[0]}")
    print(f"  lobby2: {im2.shape[1]}×{im2.shape[0]}")

    # ── Single-scale results (for comparison) ─────────────────────────────────
    print("\n══ Single-scale ════════════════════════════════════════════")
    print("\n── Set 1: lobby0 + lobby1 ──────────────────────────────────")
    H01_s, p0_s, p1_s, inl01_s, mos01_s = auto_stitch(im0, im1, label="S1-single")
    mos01_s = crop_to_content(mos01_s)

    print("\n── Set 2: lobby1 + lobby2 ──────────────────────────────────")
    H12_s, p1b_s, p2_s, inl12_s, mos12_s = auto_stitch(im1, im2, label="S2-single")
    mos12_s = crop_to_content(mos12_s)

    print("\n── Set 3 single-scale chain (pivot = lobby1) ───────────────")
    mos_full_s = crop_to_content(
        make_mosaic_3(im1, im0, im2, np.linalg.inv(H01_s), H12_s))

    # ── Multi-scale results ───────────────────────────────────────────────────
    print("\n══ Multi-scale ═════════════════════════════════════════════")
    print("\n── Set 1: lobby0 + lobby1 ──────────────────────────────────")
    (H01_m, p0_m, p1_m, inl01_m, mos01_m,
     cbys0_0, cbys0_1, ac0, lbl0, ac1, lbl1) = auto_stitch_ms(
        im0, im1, label="S1-multi")
    mos01_m = crop_to_content(mos01_m)

    print("\n── Set 2: lobby1 + lobby2 ──────────────────────────────────")
    (H12_m, p1b_m, p2_m, inl12_m, mos12_m,
     cbys1_0, cbys1_1, ac1b, lbl1b, ac2, lbl2) = auto_stitch_ms(
        im1, im2, label="S2-multi")
    mos12_m = crop_to_content(mos12_m)

    print("\n── Set 3 multi-scale chain (pivot = lobby1) ─────────────────")
    mos_full_m = crop_to_content(
        make_mosaic_3(im1, im0, im2, np.linalg.inv(H01_m), H12_m))

    # ── Corner visualizations ─────────────────────────────────────────────────
    print("\n── Saving corner images ────────────────────────────────────")
    for name, im, cbys, ac, lbl in [
        ("lobby0", im0, cbys0_0, ac0, lbl0),
        ("lobby1", im1, cbys0_1, ac1, lbl1),  # lobby1 from Set 1 perspective
        ("lobby2", im2, cbys1_1, ac2, lbl2),
    ]:
        plot_ms_corners(im, cbys, ac, lbl,
                        name=name,
                        save=f"ms_corners_{name}.png")

    # ── Match visualizations ──────────────────────────────────────────────────
    print("\n── Saving match images ─────────────────────────────────────")
    plot_lowe_matches(im0, im1, p0_m, p1_m,
                      "Set 1 multi-scale: lobby0 ↔ lobby1",
                      save="ms_lowe_01.png")
    plot_matches(im0, im1, p0_m, p1_m, inl01_m,
                 "Set 1 multi-scale: lobby0 ↔ lobby1",
                 save="ms_ransac_01.png")

    plot_lowe_matches(im1, im2, p1b_m, p2_m,
                      "Set 2 multi-scale: lobby1 ↔ lobby2",
                      save="ms_lowe_12.png")
    plot_matches(im1, im2, p1b_m, p2_m, inl12_m,
                 "Set 2 multi-scale: lobby1 ↔ lobby2",
                 save="ms_ransac_12.png")

    # ── Save individual mosaics ───────────────────────────────────────────────
    plt.imsave("ms_result_01.png",  np.clip(mos01_m,   0, 1))
    plt.imsave("ms_result_12.png",  np.clip(mos12_m,   0, 1))
    plt.imsave("ms_result_012.png", np.clip(mos_full_m, 0, 1))
    print("Saved: ms_result_01.png, ms_result_12.png, ms_result_012.png")

    # ── Comparison: single vs multi ───────────────────────────────────────────
    print("\n── Comparison figures ──────────────────────────────────────")
    plot_comparison(mos01_s,   mos01_m,   "Set 1  lobby0+lobby1",
                    save="ms_comparison_01.png")
    plot_comparison(mos12_s,   mos12_m,   "Set 2  lobby1+lobby2",
                    save="ms_comparison_12.png")
    plot_comparison(mos_full_s, mos_full_m, "Set 3  lobby0+lobby1+lobby2",
                    save="ms_comparison_012.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n══ Summary ═════════════════════════════════════════════════")
    print(f"{'':20}  {'Single-scale':>14}  {'Multi-scale':>14}")
    print("─" * 52)
    for label, inl_s, tot_s, inl_m, tot_m in [
        ("Set1 RANSAC inliers", inl01_s.sum(), len(p0_s),
         inl01_m.sum(), len(p0_m)),
        ("Set2 RANSAC inliers", inl12_s.sum(), len(p1b_s),
         inl12_m.sum(), len(p1b_m)),
    ]:
        print(f"  {label:<20}  {inl_s:>5}/{tot_s:<5}  ({inl_s/tot_s*100:4.0f}%)  "
              f"  {inl_m:>5}/{tot_m:<5}  ({inl_m/tot_m*100:4.0f}%)")


if __name__ == "__main__":
    main()
