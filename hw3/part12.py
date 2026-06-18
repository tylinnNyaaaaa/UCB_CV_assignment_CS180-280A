import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage import draw
from scipy.ndimage import map_coordinates
from PIL import Image

HW3 = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HW3, 'output')
os.makedirs(OUT, exist_ok=True)


def load_images():
    imgA = np.array(Image.open(os.path.join(HW3, 'imgA.jpg')).convert('RGB'))
    imgB_pil = Image.open(os.path.join(HW3, 'imgB.jpg')).convert('RGB')
    if (imgB_pil.height, imgB_pil.width) != imgA.shape[:2]:
        imgB_pil = imgB_pil.resize((imgA.shape[1], imgA.shape[0]))
    return imgA, np.array(imgB_pil)


def collect_keypoints(imgA, imgB):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(imgA)
    ax.set_title('Image A — click keypoints, press Enter when done')
    plt.tight_layout()
    print("Click keypoints on Image A. Press Enter when done.")
    pts_A = np.array(plt.ginput(n=-1, timeout=0)).reshape(-1, 2)
    plt.close(fig)
    if len(pts_A) == 0:
        raise RuntimeError("No keypoints collected on Image A. Please click at least one point.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(imgA)
    axes[0].set_title('Image A (reference)')
    axes[0].plot(pts_A[:, 0], pts_A[:, 1], 'r.', markersize=6)
    for i, (x, y) in enumerate(pts_A):
        axes[0].text(x + 3, y + 3, str(i), color='yellow', fontsize=7)
    axes[1].imshow(imgB)
    axes[1].set_title(f'Image B — click {len(pts_A)} matching keypoints')
    plt.tight_layout()
    print(f"Now click {len(pts_A)} matching keypoints on Image B (right panel).")
    pts_B = np.array(plt.ginput(n=len(pts_A), timeout=0)).reshape(-1, 2)
    plt.close(fig)

    return pts_A, pts_B


def add_corners(pts, shape):
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
    return np.vstack([pts, corners])


# ── Part 2 functions ─────────────────────────────────────────────────────────

def computeAffine(tri1_pts, tri2_pts):
    """
    Return 2x3 affine matrix A such that A @ [x, y, 1].T = [x', y'].T
    tri1_pts, tri2_pts: (3, 2) arrays of (x, y) vertices.
    Solve dst = A @ src  =>  A = dst @ inv(src).
    """
    ones = np.ones((3, 1))
    src = np.hstack([tri1_pts, ones]).T   # (3, 3): each column is [x, y, 1]
    dst = tri2_pts.T                       # (2, 3): rows are x', y'
    return dst @ np.linalg.inv(src)


def inverse_warp(src_img, src_pts, dst_pts, tri):
    """
    Warp src_img from src_pts geometry into dst_pts geometry.
    One loop over triangles only; pixels inside each triangle are vectorised.
    """
    h, w = src_img.shape[:2]
    n_ch = src_img.shape[2] if src_img.ndim == 3 else 1
    output = np.zeros_like(src_img, dtype=float)

    for simplex in tri.simplices:
        dst_tri = dst_pts[simplex]   # (3, 2) — (x, y) in destination space
        src_tri = src_pts[simplex]   # (3, 2) — (x, y) in source space

        # All pixel positions inside the destination triangle
        rr, cc = draw.polygon(dst_tri[:, 1], dst_tri[:, 0], shape=(h, w))
        if len(rr) == 0:
            continue

        # Inverse affine: destination coords → source coords
        A = computeAffine(dst_tri, src_tri)

        # Vectorised application of A to all (col, row) pairs at once
        coords_h = np.vstack([cc.astype(float),
                               rr.astype(float),
                               np.ones(len(rr))])   # (3, N)
        src_xy = A @ coords_h                        # (2, N): [src_x, src_y]
        src_col = src_xy[0]
        src_row = src_xy[1]

        # Interpolate source image at sub-pixel positions
        if src_img.ndim == 3:
            for c in range(n_ch):
                output[rr, cc, c] = map_coordinates(
                    src_img[:, :, c].astype(float),
                    [src_row, src_col],
                    order=1, mode='nearest'
                )
        else:
            output[rr, cc] = map_coordinates(
                src_img.astype(float),
                [src_row, src_col],
                order=1, mode='nearest'
            )

    return np.clip(output, 0, 255).astype(np.uint8)


def compute_midway_face(imgA, imgB, pts_A, pts_B, tri):
    pts_mid = (pts_A + pts_B) / 2
    warpA = inverse_warp(imgA, pts_A, pts_mid, tri)
    warpB = inverse_warp(imgB, pts_B, pts_mid, tri)
    return np.clip((warpA.astype(float) + warpB.astype(float)) / 2, 0, 255).astype(np.uint8)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true',
                        help='Ignore saved keypoints and re-collect interactively')
    args = parser.parse_args()

    imgA, imgB = load_images()

    pts_path_A = os.path.join(HW3, 'pts_A.npy')
    pts_path_B = os.path.join(HW3, 'pts_B.npy')

    has_saved = os.path.exists(pts_path_A) and os.path.exists(pts_path_B)
    if has_saved and not args.reset:
        pts_A = np.load(pts_path_A)
        pts_B = np.load(pts_path_B)
        print(f"Loaded {len(pts_A)} saved keypoints. (use --reset to re-collect)")
    else:
        pts_A_raw, pts_B_raw = collect_keypoints(imgA, imgB)
        pts_A = add_corners(pts_A_raw, imgA.shape)
        pts_B = add_corners(pts_B_raw, imgB.shape)
        np.save(pts_path_A, pts_A)
        np.save(pts_path_B, pts_B)
        print(f"Saved {len(pts_A)} keypoints (including 4 corners).")

    # ── Part 1: Triangulation on midpoints ───────────────────────────────────
    pts_mid = (pts_A + pts_B) / 2
    tri = Delaunay(pts_mid)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(imgA)
    ax.triplot(pts_mid[:, 0], pts_mid[:, 1], tri.simplices,
               color='lime', linewidth=0.8)
    ax.plot(pts_A[:, 0], pts_A[:, 1], 'r.', markersize=4)
    ax.set_title('Delaunay triangulation (midpoints) overlaid on Image A')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'triangulation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved output/triangulation.png")

    # ── Part 2: Mid-way face ──────────────────────────────────────────────────
    midway = compute_midway_face(imgA, imgB, pts_A, pts_B, tri)
    Image.fromarray(midway).save(os.path.join(OUT, 'midway_face.png'))
    print("Saved output/midway_face.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes,
                               [imgA, midway, imgB],
                               ['Image A', 'Midway Face', 'Image B']):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved output/result.png")


if __name__ == '__main__':
    main()
