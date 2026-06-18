"""
CS180 Project 3 — Part 4: Face Average
Steps 1-6: run directly   → python part4.py
Steps 8-9: cross-warp     → python part4.py --warp-me
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part12 import inverse_warp, HW3, OUT

DATA_DIR = os.path.join(HW3, 'imm_data')
IMM_W, IMM_H = 640, 480          # all IMM images are this size
N_PTS = 58                        # anatomical keypoints per face


# ── Dataset helpers ───────────────────────────────────────────────────────────

def parse_asf(path):
    """Return (58, 2) float array of pixel coords [x, y] from an .asf file."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

    # First non-comment line is the point count
    n_pts = int(lines[0])
    pts = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x_px = float(parts[2]) * IMM_W
            y_px = float(parts[3]) * IMM_H
            pts.append([x_px, y_px])
        except ValueError:
            continue
        if len(pts) == n_pts:
            break

    return np.array(pts, dtype=float)   # (58, 2)


def add_imm_corners(pts):
    """Append 4 corners of the 640×480 frame so triangulation covers the image."""
    corners = np.array([[0, 0], [IMM_W - 1, 0],
                        [0, IMM_H - 1], [IMM_W - 1, IMM_H - 1]], dtype=float)
    return np.vstack([pts, corners])


def load_imm_dataset(data_dir):
    """
    Load all 240 IMM images and their .asf keypoints.
    Returns:
        images : list of 240 np.ndarray (H, W, 3)
        pts_list: list of 240 (62, 2) arrays  [58 anatomical + 4 corners]
    """
    images, pts_list = [], []
    asf_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.asf'))

    for i, asf_name in enumerate(asf_files):
        base = asf_name[:-4]
        jpg_path = os.path.join(data_dir, base + '.jpg')
        asf_path = os.path.join(data_dir, asf_name)

        img = np.array(Image.open(jpg_path).convert('RGB'))
        pts = parse_asf(asf_path)               # (58, 2)
        pts_ext = add_imm_corners(pts)           # (62, 2)

        images.append(img)
        pts_list.append(pts_ext)

        print(f'\rLoading dataset {i + 1}/{len(asf_files)}', end='', flush=True)

    print()
    return images, pts_list


# ── Core computation ──────────────────────────────────────────────────────────

def compute_mean_shape(pts_list):
    """Mean of all (62, 2) arrays → (62, 2)."""
    return np.mean(np.stack(pts_list, axis=0), axis=0)


def build_mean_face(images, pts_list, mean_pts, tri):
    """
    Warp every image to mean_pts, accumulate, and average.
    Returns uint8 (H, W, 3).
    """
    n = len(images)
    acc = np.zeros((IMM_H, IMM_W, 3), dtype=float)

    for i, (img, pts) in enumerate(zip(images, pts_list)):
        print(f'\rWarping image {i + 1}/{n}', end='', flush=True)
        warped = inverse_warp(img, pts, mean_pts, tri)
        acc += warped.astype(float)

    print()
    return np.clip(acc / n, 0, 255).astype(np.uint8)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def save_reference_image(images, pts_list, path):
    """
    Overlay the 58 IMM keypoint indices on an example face and save.
    Uses the first image in the dataset.
    """
    img = images[0]
    pts = pts_list[0][:N_PTS]              # only the 58 anatomical pts

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.plot(pts[:, 0], pts[:, 1], 'r.', markersize=4)
    for idx, (x, y) in enumerate(pts):
        ax.text(x + 3, y - 3, str(idx), color='yellow',
                fontsize=6, fontweight='bold')
    ax.set_title('IMM 58-point reference (image 01-1m)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def save_examples_grid(images, pts_list, mean_pts, tri, path, n_show=8):
    """
    Show n_show original faces alongside their warp-to-mean result.
    """
    indices = np.linspace(0, len(images) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    fig.suptitle('Original (top) vs warped to mean shape (bottom)', fontsize=11)

    for col, idx in enumerate(indices):
        img = images[idx]
        warped = inverse_warp(img, pts_list[idx], mean_pts, tri)

        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        axes[1, col].imshow(warped)
        axes[1, col].axis('off')

        print(f'\rRendering example {col + 1}/{n_show}', end='', flush=True)

    print()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ── Warp-me step (steps 8-9) ─────────────────────────────────────────────────

def warp_me(mean_pts, tri, mean_face_img, images, pts_list):
    """Use dataset image 0 as stand-in, compute both cross-warps, save results."""
    my_img = images[0]
    my_pts = pts_list[0]        # already (62, 2) with corners

    print('Warping "my" face (dataset img 0) → mean geometry …')
    my_to_mean = inverse_warp(my_img, my_pts, mean_pts, tri)

    print('Warping mean face → "my" geometry …')
    mean_to_my = inverse_warp(mean_face_img, mean_pts, my_pts, tri)

    Image.fromarray(my_to_mean).save(os.path.join(OUT, 'part4_my_to_mean.png'))
    Image.fromarray(mean_to_my).save(os.path.join(OUT, 'part4_mean_to_my.png'))
    print('Saved part4_my_to_mean.png, part4_mean_to_my.png')

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, img, title in zip(
        axes,
        [my_img, mean_face_img, my_to_mean, mean_to_my],
        ['"My" face (dataset img 0)', 'Mean face',
         '"My" → mean shape', 'Mean → "my" shape'],
    ):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'part4_result.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved part4_result.png')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warp-me', action='store_true',
                        help='Cross-warp dataset img 0 ↔ mean face')
    args = parser.parse_args()

    # ── Steps 1–6: population mean (always run first) ────────────────────────

    mean_face_path = os.path.join(OUT, 'part4_mean_face.png')
    mean_pts_path  = os.path.join(HW3, 'mean_pts_62.npy')

    images, pts_list = load_imm_dataset(DATA_DIR)

    mean_pts = compute_mean_shape(pts_list)        # (62, 2)
    np.save(mean_pts_path, mean_pts)

    tri = Delaunay(mean_pts)

    if not os.path.exists(mean_face_path):
        mean_face = build_mean_face(images, pts_list, mean_pts, tri)
        Image.fromarray(mean_face).save(mean_face_path)
        print(f'Saved {mean_face_path}')
    else:
        mean_face = np.array(Image.open(mean_face_path).convert('RGB'))
        print('Loaded cached mean face.')

    ref_path = os.path.join(OUT, 'imm_point_reference.png')
    save_reference_image(images, pts_list, ref_path)

    examples_path = os.path.join(OUT, 'part4_examples.png')
    save_examples_grid(images, pts_list, mean_pts, tri, examples_path, n_show=8)

    # ── Steps 8–9: cross-warp with dataset img 0 ─────────────────────────────
    if args.warp_me:
        warp_me(mean_pts, tri, mean_face, images, pts_list)
        return


if __name__ == '__main__':
    main()