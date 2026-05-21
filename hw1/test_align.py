import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.transform import resize
from scipy.ndimage import shift

img = io.imread('img.tif')
if img.dtype == np.uint16:
    img = img.astype(np.float64) / 65535.0
else:
    img = img.astype(np.float64) / 255.0

height = img.shape[0]
third = height // 3
b = img[:third]
g = img[third:2*third]
r = img[2*third:3*third]


def ncc(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return np.sum(a * b)

def edge_feature(img):
    gx = filters.sobel_v(img)
    gy = filters.sobel_h(img)
    return np.sqrt(gx**2 + gy**2)

def align_pyramid(base, to_align, feature_fn, min_size=32, window=15):
    levels, h = [], base.shape[0]
    while h > min_size:
        levels.append(h)
        h //= 2
    levels = levels[::-1]  # coarse → fine

    best_dx, best_dy = 0, 0
    for level_h in levels:
        scale = level_h / base.shape[0]
        w = int(base.shape[1] * scale)
        base_s  = feature_fn(resize(base,     (level_h, w)))
        align_s = feature_fn(resize(to_align, (level_h, w)))

        sdx, sdy = best_dx * 2, best_dy * 2
        search_window = window if level_h == levels[0] else 2
        margin = max(5, level_h // 8)

        best_score = -np.inf
        for dy in range(sdy - search_window, sdy + search_window + 1):
            for dx in range(sdx - search_window, sdx + search_window + 1):
                shifted = shift(align_s, [dy, dx], mode='constant', cval=0)
                score = ncc(base_s[margin:-margin, margin:-margin],
                            shifted[margin:-margin, margin:-margin])
                if score > best_score:
                    best_score = score
                    best_dx, best_dy = dx, dy

    return best_dx, best_dy

def compose(b, g, r, dx_g, dy_g, dx_r, dy_r):
    g_al = shift(g, [dy_g, dx_g], mode='constant', cval=0)
    r_al = shift(r, [dy_r, dx_r], mode='constant', cval=0)
    color = np.stack([r_al, g_al, b], axis=2)

    # 第一段：根據 offset 精確裁掉 zero-fill 黑邊
    h, w = color.shape[:2]
    dy_max, dx_max = max(abs(dy_g), abs(dy_r)), max(abs(dx_g), abs(dx_r))
    top    = dy_max if dy_max > 0 else 0
    bottom = h + min(dy_g, dy_r, 0)   # 負 offset 時下方有黑邊
    left   = dx_max if dx_max > 0 else 0
    right  = w + min(dx_g, dx_r, 0)   # 負 offset 時右方有黑邊
    return color[top:bottom, left:right]

def auto_crop(img, margin=0.15):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    h, w = img.shape[:2]
    search_h = int(h * margin)
    search_w = int(w * margin)

    top, bottom = 0, h
    for ch in [r, g, b]:
        row_edge = np.abs(filters.sobel_h(ch)).mean(axis=1)
        ch_top    = int(np.argmax(row_edge[:search_h])) + 1
        ch_bottom = h - 1 - int(np.argmax(row_edge[h - search_h:][::-1]))
        top    = max(top,    ch_top)
        bottom = min(bottom, ch_bottom)

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


# 原始像素（baseline）
print("=== 原始像素對齊 ===")
dx_g0, dy_g0 = align_pyramid(b, g, feature_fn=lambda x: x)
dx_r0, dy_r0 = align_pyramid(b, r, feature_fn=lambda x: x)
print(f"G: dx={dx_g0}, dy={dy_g0}  R: dx={dx_r0}, dy={dy_r0}")
result_raw      = compose(b, g, r, dx_g0, dy_g0, dx_r0, dy_r0)
result_raw_crop = auto_crop(result_raw)

# Sobel 梯度大小
print("\n=== Sobel 梯度特徵對齊 ===")
dx_g1, dy_g1 = align_pyramid(b, g, feature_fn=edge_feature)
dx_r1, dy_r1 = align_pyramid(b, r, feature_fn=edge_feature)
print(f"G: dx={dx_g1}, dy={dy_g1}  R: dx={dx_r1}, dy={dy_r1}")
result_edge      = compose(b, g, r, dx_g1, dy_g1, dx_r1, dy_r1)
result_edge_crop = auto_crop(result_edge)


fig, axes = plt.subplots(2, 2, figsize=(16, 16))

axes[0, 0].imshow(np.clip(result_raw, 0, 1))
axes[0, 0].set_title(f'原始像素（合成）\nG:({dx_g0},{dy_g0}) R:({dx_r0},{dy_r0})')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.clip(result_raw_crop, 0, 1))
axes[0, 1].set_title('原始像素（auto_crop 後）')
axes[0, 1].axis('off')

axes[1, 0].imshow(np.clip(result_edge, 0, 1))
axes[1, 0].set_title(f'Sobel 梯度特徵（合成）\nG:({dx_g1},{dy_g1}) R:({dx_r1},{dy_r1})')
axes[1, 0].axis('off')

axes[1, 1].imshow(np.clip(result_edge_crop, 0, 1))
axes[1, 1].set_title('Sobel 梯度特徵（auto_crop 後）')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('test_align_compare.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("已儲存 test_align_compare.jpg")
