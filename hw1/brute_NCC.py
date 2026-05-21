import numpy as np
from skimage import io
import matplotlib.pyplot as plt

img = io.imread('img.jpg')
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

def align(img_base, img_to_align, window=15):
    best_score = -np.inf
    best_dx, best_dy = 0, 0
    
    for dy in range(-window, window+1):
        for dx in range(-window, window+1):
            # 位移圖片
            shifted = np.roll(img_to_align, dy, axis=0)
            shifted = np.roll(shifted, dx, axis=1)
            
            # 只比較內部像素（去掉邊緣）
            score = ncc(img_base[10:-10, 10:-10], shifted[10:-10, 10:-10])
            
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
    
    return best_dx, best_dy

dx_g, dy_g = align(b, g)
dx_r, dy_r = align(b, r)

print(f"G offset: dx={dx_g}, dy={dy_g}")
print(f"R offset: dx={dx_r}, dy={dy_r}")

g_aligned = np.roll(g, dy_g, axis=0)
g_aligned = np.roll(g_aligned, dx_g, axis=1)

r_aligned = np.roll(r, dy_r, axis=0)
r_aligned = np.roll(r_aligned, dx_r, axis=1)

color_img = np.stack([r_aligned, g_aligned, b], axis=2)
plt.imsave('output_brute.jpg', color_img)
print("已儲存 output_brute.jpg")