"""
CS180 Project 3 — Bells & Whistles: Gender Change
Morphs male ↔ female using IMM gender-specific mean faces.
Three modes: shape only / appearance only / both.

Usage: python part5_gender.py
"""
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
from part4 import (load_imm_dataset, compute_mean_shape, build_mean_face,
                   DATA_DIR)
from part5 import split_by_gender

MALE_MEAN_IMG_PATH   = os.path.join(OUT, 'male_mean_face.png')
FEMALE_MEAN_IMG_PATH = os.path.join(OUT, 'female_mean_face.png')


# ── Three warp modes ──────────────────────────────────────────────────────────

def shape_only(src_img, src_pts, dst_mean_pts, tri):
    """Source texture warped into target mean shape."""
    return inverse_warp(src_img, src_pts, dst_mean_pts, tri)


def appearance_only(src_pts, dst_mean_img, dst_mean_pts, tri):
    """Target mean texture warped into source shape (keeps source geometry)."""
    return inverse_warp(dst_mean_img, dst_mean_pts, src_pts, tri)


def both(src_img, src_pts, dst_mean_img, dst_mean_pts, tri):
    """Full midway morph: halfway in shape and appearance."""
    pts_mid = (src_pts + dst_mean_pts) / 2
    warp_src = inverse_warp(src_img,      src_pts,      pts_mid, tri)
    warp_dst = inverse_warp(dst_mean_img, dst_mean_pts, pts_mid, tri)
    return np.clip((warp_src.astype(float) + warp_dst.astype(float)) / 2,
                   0, 255).astype(np.uint8)


# ── Save comparison grid ──────────────────────────────────────────────────────

def save_gender_grid(src_img, src_pts,
                     dst_mean_img, dst_mean_pts,
                     tri, direction, path):
    """
    4-panel: Original | Shape only | Appearance only | Both
    """
    print(f'[{direction}] shape only …')
    im_shape = shape_only(src_img, src_pts, dst_mean_pts, tri)
    print(f'[{direction}] appearance only …')
    im_appearance = appearance_only(src_pts, dst_mean_img, dst_mean_pts, tri)
    print(f'[{direction}] both …')
    im_both = both(src_img, src_pts, dst_mean_img, dst_mean_pts, tri)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [src_img, im_shape, im_appearance, im_both],
        ['Original', 'Shape only', 'Appearance only', 'Both'],
    ):
        ax.imshow(img);  ax.set_title(title);  ax.axis('off')
    fig.suptitle(direction, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    images, pts_list = load_imm_dataset(DATA_DIR)
    male_imgs, male_pts, female_imgs, female_pts = \
        split_by_gender(images, pts_list, DATA_DIR)

    male_mean_pts   = compute_mean_shape(male_pts)
    female_mean_pts = compute_mean_shape(female_pts)
    overall_mean_pts = compute_mean_shape(pts_list)

    # Triangulation on overall mean — good middle ground for both genders
    tri = Delaunay(overall_mean_pts)

    # Build (or load cached) gender mean face images
    if os.path.exists(MALE_MEAN_IMG_PATH):
        male_mean_img = np.array(Image.open(MALE_MEAN_IMG_PATH).convert('RGB'))
        print('Loaded cached male mean face.')
    else:
        print('Building male mean face …')
        male_mean_img = build_mean_face(male_imgs, male_pts, male_mean_pts,
                                        Delaunay(male_mean_pts))
        Image.fromarray(male_mean_img).save(MALE_MEAN_IMG_PATH)
        print(f'Saved {MALE_MEAN_IMG_PATH}')

    if os.path.exists(FEMALE_MEAN_IMG_PATH):
        female_mean_img = np.array(Image.open(FEMALE_MEAN_IMG_PATH).convert('RGB'))
        print('Loaded cached female mean face.')
    else:
        print('Building female mean face …')
        female_mean_img = build_mean_face(female_imgs, female_pts, female_mean_pts,
                                          Delaunay(female_mean_pts))
        Image.fromarray(female_mean_img).save(FEMALE_MEAN_IMG_PATH)
        print(f'Saved {FEMALE_MEAN_IMG_PATH}')

    # Subject faces: images[0] = 01-1m (male), images[42] = 08-1f (female)
    my_male_img   = images[0];    my_male_pts   = pts_list[0]
    my_female_img = images[42];   my_female_pts = pts_list[42]

    # Male → Female
    save_gender_grid(
        my_male_img, my_male_pts,
        female_mean_img, female_mean_pts,
        tri,
        direction='Male → Female',
        path=os.path.join(OUT, 'part5_gender_male_to_female.png'),
    )

    # Female → Male
    save_gender_grid(
        my_female_img, my_female_pts,
        male_mean_img, male_mean_pts,
        tri,
        direction='Female → Male',
        path=os.path.join(OUT, 'part5_gender_female_to_male.png'),
    )


if __name__ == '__main__':
    main()