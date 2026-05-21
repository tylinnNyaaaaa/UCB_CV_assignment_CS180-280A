import numpy as np
from skimage import io
from skimage.transform import resize
from skimage import filters
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

def auto_crop(img, margin=0.15):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    h, w = img.shape[:2]
    search_h = int(h * margin)
    search_w = int(w * margin)

    top, bottom = 0, h

    # 上下：每個 channel 的 sobel_h 峰值（水平板框邊界最強）
    for ch in [r, g, b]:
        row_edge = np.abs(filters.sobel_h(ch)).mean(axis=1)
        ch_top    = int(np.argmax(row_edge[:search_h])) + 1
        ch_bottom = h - 1 - int(np.argmax(row_edge[h - search_h:][::-1]))
        top    = max(top,    ch_top)
        bottom = min(bottom, ch_bottom)

    # 左右：inter-channel 差異（偵測 np.roll 水平偏移造成的 channel 錯位）
    diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    col_diff = diff.mean(axis=0)
    interior = col_diff[search_w : w - search_w]
    col_threshold = interior.mean() + 2 * interior.std()

    left = 0
    for i in range(search_w):
        if col_diff[i] > col_threshold:
            left = i + 1

    right = w
    for i in range(w - 1, w - 1 - search_w, -1):
        if col_diff[i] > col_threshold:
            right = i

    print(f"裁切範圍: top={top}, bottom={bottom}, left={left}, right={right}")
    return img[top:bottom, left:right]

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
color_img_cropped = auto_crop(color_img)

img_min = color_img_cropped.min()
img_max = color_img_cropped.max()
color_img_cropped = (color_img_cropped - img_min) / (img_max - img_min)

# 自動白平衡：Gray World 假設（場景平均色應為灰色，以此估計光源）
r_ch = color_img_cropped[:, :, 0]
g_ch = color_img_cropped[:, :, 1]
b_ch = color_img_cropped[:, :, 2]

gray_mean = (r_ch.mean() + g_ch.mean() + b_ch.mean()) / 3
r_wb = np.clip(r_ch * (gray_mean / r_ch.mean()), 0, 1)
g_wb = np.clip(g_ch * (gray_mean / g_ch.mean()), 0, 1)
b_wb = np.clip(b_ch * (gray_mean / b_ch.mean()), 0, 1)

color_img_wb = np.stack([r_wb, g_wb, b_wb], axis=2)
plt.imsave('output_wb.jpg', color_img_wb)
print("已儲存 output_wb.jpg")