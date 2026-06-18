"""
CS180 Project 3 — Part 5: Caricatures
Usage: python part5.py
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
from part4 import (load_imm_dataset, compute_mean_shape,
                   DATA_DIR, IMM_W, IMM_H)


# ── Helpers ───────────────────────────────────────────────────────────────────

def gender_from_name(asf_or_jpg_name):
    """Return 'm' or 'f' from filenames like '01-1m.asf' or '08-3f.jpg'."""
    base = os.path.splitext(os.path.basename(asf_or_jpg_name))[0]
    return base[-1]   # last char of base name


def split_by_gender(images, pts_list, data_dir):
    """
    Return two parallel lists: one for males, one for females.
    File order in load_imm_dataset matches sorted .asf filenames.
    """
    asf_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.asf'))
    assert len(asf_files) == len(images), "Dataset size mismatch"

    male_imgs, male_pts = [], []
    female_imgs, female_pts = [], []

    for i, asf_name in enumerate(asf_files):
        if gender_from_name(asf_name) == 'm':
            male_imgs.append(images[i])
            male_pts.append(pts_list[i])
        else:
            female_imgs.append(images[i])
            female_pts.append(pts_list[i])

    return male_imgs, male_pts, female_imgs, female_pts


# ── Core caricature function ──────────────────────────────────────────────────

def make_caricature(img, pts, ref_mean_pts, tri, alpha):
    """
    Extrapolate from ref_mean_pts through pts by factor alpha.

    delta      = pts - ref_mean_pts
    target_pts = pts + alpha * delta
    """
    delta = pts - ref_mean_pts
    target_pts = ref_mean_pts + alpha * delta
    return inverse_warp(img, pts, target_pts, tri)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_subject(label, img, pts, male_mean_pts, female_mean_pts,
                overall_mean_pts, tri, alphas):
    """
    Generate alpha-sweep grid and 3-way mean-comparison for one subject.
    Saves:
      part5_{label}_alphas.png
      part5_{label}_mean_comparison.png
    """
    # Alpha sweep vs male mean
    caricatures = []
    for a in alphas:
        print(f'\r[{label}] Generating caricature alpha={a}', end='', flush=True)
        caricatures.append(make_caricature(img, pts, male_mean_pts, tri, a))
    print()

    n_cols = 1 + len(alphas)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
    axes[0].imshow(img);  axes[0].set_title('Original');  axes[0].axis('off')
    for ax, caric, a in zip(axes[1:], caricatures, alphas):
        ax.imshow(caric);  ax.set_title(f'α = {a}');  ax.axis('off')
    fig.suptitle(f'[{label}] Caricatures — male-mean reference', fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, f'part5_{label}_alphas.png')
    plt.savefig(out, dpi=150, bbox_inches='tight');  plt.close()
    print(f'Saved {out}')

    # 3-way mean comparison at the largest alpha
    cmp_alpha = alphas[-1]
    caric_male    = caricatures[alphas.index(cmp_alpha)]
    caric_female  = make_caricature(img, pts, female_mean_pts,  tri, cmp_alpha)
    caric_overall = make_caricature(img, pts, overall_mean_pts, tri, cmp_alpha)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, im, title in zip(
        axes,
        [img, caric_male, caric_female, caric_overall],
        ['Original',
         f'Male-mean ref (α={cmp_alpha})',
         f'Female-mean ref (α={cmp_alpha})',
         f'Overall-mean ref (α={cmp_alpha})'],
    ):
        ax.imshow(im);  ax.set_title(title, fontsize=9);  ax.axis('off')
    fig.suptitle(f'[{label}] Effect of reference mean', fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, f'part5_{label}_mean_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight');  plt.close()
    print(f'Saved {out}')


def main():
    # 1. Load full dataset
    images, pts_list = load_imm_dataset(DATA_DIR)

    # 2. Split by gender; compute per-gender and overall mean shapes
    male_imgs, male_pts, female_imgs, female_pts = \
        split_by_gender(images, pts_list, DATA_DIR)
    print(f'Male: {len(male_imgs)}  Female: {len(female_imgs)}')

    male_mean_pts    = compute_mean_shape(male_pts)
    female_mean_pts  = compute_mean_shape(female_pts)
    overall_mean_pts = compute_mean_shape(pts_list)

    # 3. Shared Delaunay on male mean
    tri = Delaunay(male_mean_pts)

    alphas = [-0.5, 0.5, 1, 2]

    # ── Male subject: images[0] = 01-1m ──────────────────────────────────────
    run_subject('male', images[0], pts_list[0],
                male_mean_pts, female_mean_pts, overall_mean_pts, tri, alphas)

    # ── Female subject: images[42] = 08-1f ───────────────────────────────────
    run_subject('female', images[42], pts_list[42],
                male_mean_pts, female_mean_pts, overall_mean_pts, tri, alphas)


if __name__ == '__main__':
    main()