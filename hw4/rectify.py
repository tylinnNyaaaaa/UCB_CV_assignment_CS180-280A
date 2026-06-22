"""
Image Rectification using Homography
Rectify the rectangular mat in img3.jpg to a frontal view.

Usage:
    python rectify.py              # click 4 corners interactively (default)
    python rectify.py --hardcoded  # use pre-saved hardcoded corners
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread as mpl_imread
from homography import computeH, warpImage

matplotlib.use('TkAgg')


# ---------------------------------------------------------------------------
# Pre-saved corners (fallback / reference)  (x, y) pixel coords  1477×1108
# Order: top-left, top-right, bottom-right, bottom-left
# ---------------------------------------------------------------------------
MAT_CORNERS = np.array([
    [ 596,  433],
    [1159,  425],
    [1289,  776],
    [ 374,  781],
], dtype=np.float64)

OUT_W, OUT_H = 600, 450


def pick_corners(im: np.ndarray, n: int = 4) -> np.ndarray:
    """Display image and let user click n corners; return (n,2) array."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(im)
    ax.set_title(
        f"Click the {n} corners of the rectangular object\n"
        "Order: top-left → top-right → bottom-right → bottom-left",
        fontsize=11,
    )
    pts = plt.ginput(n, timeout=120)
    plt.close(fig)
    return np.array(pts, dtype=np.float64)   # (n, 2)


def rectify(im: np.ndarray, src_pts: np.ndarray,
            out_w: int, out_h: int) -> np.ndarray:
    """
    Compute homography from src_pts quad → output rectangle,
    warp the image, and extract the rectangular result.

    warpImage positions the canvas at (x_min, y_min) of the full mapped
    bounding box, so we need to shift by that offset to find (0,0) in
    destination space (i.e. the top-left of our target rectangle).
    """
    dst_pts = np.array([
        [0,     0    ],
        [out_w, 0    ],
        [out_w, out_h],
        [0,     out_h],
    ], dtype=np.float64)

    H = computeH(src_pts, dst_pts)

    # Determine canvas offset that warpImage will apply
    h_im, w_im = im.shape[:2]
    corners_hom = np.array([[0, 0, 1], [w_im, 0, 1],
                             [0, h_im, 1], [w_im, h_im, 1]],
                            dtype=np.float64).T   # (3,4)
    proj = H @ corners_hom
    proj /= proj[2:3]
    x_min = proj[0].min()
    y_min = proj[1].min()

    warped, _ = warpImage(im, H)

    # Top-left of target rect sits at (0-x_min, 0-y_min) in the canvas
    r0 = max(0, int(round(-y_min)))
    c0 = max(0, int(round(-x_min)))
    r1 = r0 + out_h
    c1 = c0 + out_w

    return np.clip(warped[r0:r1, c0:c1], 0.0, 1.0)


def main():
    hardcoded = "--hardcoded" in sys.argv

    im = mpl_imread("img3.jpg").astype(np.float64) / 255.0

    if hardcoded:
        src_pts = MAT_CORNERS
        print("Using hardcoded mat corners:")
        labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
        for label, (x, y) in zip(labels, src_pts):
            print(f"  {label}: ({x:.0f}, {y:.0f})")
        out_w, out_h = OUT_W, OUT_H
    else:
        print("Click the 4 corners of the rectangular object in order:")
        print("  top-left → top-right → bottom-right → bottom-left")
        src_pts = pick_corners(im, n=4)
        print("\nSelected corners (x, y):")
        labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
        for label, (x, y) in zip(labels, src_pts):
            print(f"  {label}: ({x:.1f}, {y:.1f})")

        # Estimate output size from the clicked quad
        top_w  = np.linalg.norm(src_pts[1] - src_pts[0])
        bot_w  = np.linalg.norm(src_pts[2] - src_pts[3])
        left_h = np.linalg.norm(src_pts[3] - src_pts[0])
        rgt_h  = np.linalg.norm(src_pts[2] - src_pts[1])
        out_w  = int(round((top_w + bot_w) / 2))
        out_h  = int(round((left_h + rgt_h) / 2))
        print(f"\nAuto output size: {out_w} × {out_h}")

    rectified = rectify(im, src_pts, out_w, out_h)

    # --- Visualise ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Image Rectification — img3.jpg", fontsize=14, fontweight="bold")

    ax0 = axes[0]
    ax0.imshow(im)
    # Draw the selected quad
    poly = np.vstack([src_pts, src_pts[0]])   # close the loop
    ax0.plot(poly[:, 0], poly[:, 1], "r-o", linewidth=2, markersize=6)
    for label, pt in zip(["TL", "TR", "BR", "BL"], src_pts):
        ax0.text(pt[0] + 8, pt[1] - 8, label,
                 color="yellow", fontsize=10, fontweight="bold")
    ax0.set_title(f"Original  ({im.shape[1]}×{im.shape[0]})", fontsize=11)
    ax0.axis("off")

    ax1 = axes[1]
    ax1.imshow(rectified)
    ax1.set_title(f"Rectified  ({out_w}×{out_h})", fontsize=11)
    ax1.axis("off")

    plt.tight_layout()
    out_path = "rectified_img3.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
