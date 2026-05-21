import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

img = io.imread('img.tif')
print("資料型別:", img.dtype)  # 確認是 uint16

# uint16 範圍是 0~65535
if img.dtype == np.uint16:
    img = img.astype(np.float64) / 65535.0
else:
    img = img.astype(np.float64) / 255.0

height = img.shape[0]
third = height // 3
b = img[:third]
g = img[third:2*third]
r = img[2*third:3*third]

def ncc(img1, img2):
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)
    return np.sum(img1_norm * img2_norm)

def align_pyramid(img_base, img_to_align, min_size=32, window=15):
    levels = []
    h = img_base.shape[0]
    while h > min_size:
        levels.append(h)
        h = h // 2
    levels = levels[::-1]

    print(f"金字塔層數: {len(levels)}, 各層高度: {levels}")

    best_dx, best_dy = 0, 0

    for level_h in levels:
        scale = level_h / img_base.shape[0]
        w = int(img_base.shape[1] * scale)

        base_small = resize(img_base, (level_h, w))
        align_small = resize(img_to_align, (level_h, w))

        start_dx = best_dx * 2
        start_dy = best_dy * 2

        best_score = -np.inf

        for dy in range(start_dy - window, start_dy + window + 1):
            for dx in range(start_dx - window, start_dx + window + 1):
                shifted = np.roll(align_small, dy, axis=0)
                shifted = np.roll(shifted, dx, axis=1)
                score = ncc(base_small[10:-10, 10:-10], shifted[10:-10, 10:-10])

                if score > best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy

        print(f"層高度 {level_h}: dx={best_dx}, dy={best_dy}")

    return best_dx, best_dy

dx_g, dy_g = align_pyramid(b, g)
dx_r, dy_r = align_pyramid(b, r)

print(f"\n最終結果:")
print(f"G offset: dx={dx_g}, dy={dy_g}")
print(f"R offset: dx={dx_r}, dy={dy_r}")

g_aligned = np.roll(g, dy_g, axis=0)
g_aligned = np.roll(g_aligned, dx_g, axis=1)
r_aligned = np.roll(r, dy_r, axis=0)
r_aligned = np.roll(r_aligned, dx_r, axis=1)

color_img = np.stack([r_aligned, g_aligned, b], axis=2)
plt.imsave('output_pyramid_tif.jpg', color_img)
print("已儲存 output_pyramid_tif.jpg")