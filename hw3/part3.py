import os
import sys
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part12 import load_images, inverse_warp, HW3, OUT


def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    """
    Produce one morphed frame between im1 and im2.

    warp_frac    : 0 → keep im1 shape,  1 → keep im2 shape
    dissolve_frac: 0 → show im1 pixels, 1 → show im2 pixels
    """
    # Intermediate point set (same triangulation connectivity, shifted vertices)
    pts_warp = (1.0 - warp_frac) * im1_pts + warp_frac * im2_pts

    warp1 = inverse_warp(im1, im1_pts, pts_warp, tri)
    warp2 = inverse_warp(im2, im2_pts, pts_warp, tri)

    blended = (1.0 - dissolve_frac) * warp1.astype(float) \
            +        dissolve_frac  * warp2.astype(float)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_morph_sequence(imgA, imgB, pts_A, pts_B, tri,
                             n_frames=45, fps=30):
    """
    Generate n_frames frames with warp_frac == dissolve_frac going 0→1.
    Returns list of PIL Images.
    """
    frames = []
    for i, t in enumerate(np.linspace(0, 1, n_frames)):
        print(f"\rGenerating frame {i + 1}/{n_frames}  (t={t:.3f})", end='', flush=True)
        frame = morph(imgA, imgB, pts_A, pts_B, tri,
                      warp_frac=t, dissolve_frac=t)
        frames.append(Image.fromarray(frame))
    print()
    return frames


def save_gif(frames, path, fps=30):
    ms_per_frame = int(1000 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=ms_per_frame,
        optimize=False,
    )
    print(f"Saved {path}  ({len(frames)} frames @ {fps} fps)")


def main():
    imgA, imgB = load_images()

    pts_path_A = os.path.join(HW3, 'pts_A.npy')
    pts_path_B = os.path.join(HW3, 'pts_B.npy')
    if not (os.path.exists(pts_path_A) and os.path.exists(pts_path_B)):
        raise FileNotFoundError(
            "Run part12.py first to collect and save keypoints.")

    pts_A = np.load(pts_path_A)
    pts_B = np.load(pts_path_B)

    # Triangulation on midpoints (same structure used across all frames)
    pts_mid = (pts_A + pts_B) / 2
    tri = Delaunay(pts_mid)

    frames = generate_morph_sequence(imgA, imgB, pts_A, pts_B, tri,
                                     n_frames=45, fps=30)

    save_gif(frames, os.path.join(OUT, 'morph.gif'), fps=30)


if __name__ == '__main__':
    main()