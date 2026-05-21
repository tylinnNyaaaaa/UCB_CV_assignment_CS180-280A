import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters

# 直接用已對齊的圖（pyramid.py 的輸出），和 optimize.py 的輸入情境相同
img = io.imread('output_pyramid_tif.jpg').astype(np.float64) / 255.0
h, w = img.shape[:2]
r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
margin = 0.15
search_h = int(h * margin)
search_w = int(w * margin)


def find_tb_sobel_h(channel, search_h, h):
    row_edge = np.abs(filters.sobel_h(channel)).mean(axis=1)
    top    = int(np.argmax(row_edge[:search_h])) + 1
    bottom = h - 1 - int(np.argmax(row_edge[h - search_h:][::-1]))
    return top, bottom

def find_lr_sobel_v(channel, search_w, w):
    col_edge = np.abs(filters.sobel_v(channel)).mean(axis=0)
    left  = int(np.argmax(col_edge[:search_w])) + 1
    right = w - 1 - int(np.argmax(col_edge[w - search_w:][::-1]))
    return left, right

def find_lr_interchannel(r, g, b, search_w, w):
    diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    col_diff = diff.mean(axis=0)
    interior = col_diff[search_w : w - search_w]
    thr = interior.mean() + 2 * interior.std()
    left = 0
    for i in range(search_w):
        if col_diff[i] > thr:
            left = i + 1
    right = w
    for i in range(w - 1, w - 1 - search_w, -1):
        if col_diff[i] > thr:
            right = i
    return left, right

def find_tb_lr_magnitude(channel, search_h, search_w, h, w):
    edge = filters.sobel(channel)
    row_edge = edge.mean(axis=1)
    col_edge = edge.mean(axis=0)
    top    = int(np.argmax(row_edge[:search_h])) + 1
    bottom = h - 1 - int(np.argmax(row_edge[h - search_h:][::-1]))
    left   = int(np.argmax(col_edge[:search_w])) + 1
    right  = w - 1 - int(np.argmax(col_edge[w - search_w:][::-1]))
    return top, bottom, left, right


# Method 1: Sobel_h (上下) + Sobel_v (左右)
top1, bottom1, left1, right1 = 0, h, 0, w
for ch in [r, g, b]:
    t, bo = find_tb_sobel_h(ch, search_h, h)
    l, ri = find_lr_sobel_v(ch, search_w, w)
    top1    = max(top1,    t)
    bottom1 = min(bottom1, bo)
    left1   = max(left1,   l)
    right1  = min(right1,  ri)
crop1 = img[top1:bottom1, left1:right1]
print(f"M1 (sobel_h + sobel_v):     top={top1}, bottom={bottom1}, left={left1}, right={right1}")

# Method 2: Sobel_h (上下) + Inter-channel (左右)
top2, bottom2 = 0, h
for ch in [r, g, b]:
    t, bo = find_tb_sobel_h(ch, search_h, h)
    top2    = max(top2,    t)
    bottom2 = min(bottom2, bo)
left2, right2 = find_lr_interchannel(r, g, b, search_w, w)
crop2 = img[top2:bottom2, left2:right2]
print(f"M2 (sobel_h + interchannel): top={top2}, bottom={bottom2}, left={left2}, right={right2}")

# Method 3: Sobel magnitude (全方向)
top3, bottom3, left3, right3 = 0, h, 0, w
for ch in [r, g, b]:
    t, bo, l, ri = find_tb_lr_magnitude(ch, search_h, search_w, h, w)
    top3    = max(top3,    t)
    bottom3 = min(bottom3, bo)
    left3   = max(left3,   l)
    right3  = min(right3,  ri)
crop3 = img[top3:bottom3, left3:right3]
print(f"M3 (sobel magnitude):        top={top3}, bottom={bottom3}, left={left3}, right={right3}")


fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes[0].imshow(img)
axes[0].set_title('原圖')
axes[0].axis('off')

axes[1].imshow(crop1)
axes[1].set_title(f'M1: sobel_h + sobel_v\n({crop1.shape[1]}x{crop1.shape[0]})')
axes[1].axis('off')

axes[2].imshow(crop2)
axes[2].set_title(f'M2: sobel_h + interchannel\n({crop2.shape[1]}x{crop2.shape[0]})')
axes[2].axis('off')

axes[3].imshow(crop3)
axes[3].set_title(f'M3: sobel magnitude\n({crop3.shape[1]}x{crop3.shape[0]})')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('test_crop_compare.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("已儲存 test_crop_compare.jpg")
