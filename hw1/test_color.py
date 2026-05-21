import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

img = io.imread('output_wb.jpg').astype(np.float64) / 255.0


def stretch(channel, lo_pct=1, hi_pct=99):
    lo = np.percentile(channel, lo_pct)
    hi = np.percentile(channel, hi_pct)
    return np.clip((channel - lo) / (hi - lo), 0, 1)


# Method 1: HSV — 只拉伸 V（亮度）
hsv = color.rgb2hsv(img)
hsv[:, :, 2] = stretch(hsv[:, :, 2])
result_hsv = color.hsv2rgb(hsv)

# Method 2: HSV — 拉伸 V，微調 S 增加飽和度
hsv2 = color.rgb2hsv(img)
hsv2[:, :, 2] = stretch(hsv2[:, :, 2])
hsv2[:, :, 1] = np.clip(hsv2[:, :, 1] * 1.3, 0, 1)
result_hsv_sat = color.hsv2rgb(hsv2)

# Method 3: YUV — 只拉伸 Y（亮度）
yuv = color.rgb2yuv(img)
yuv[:, :, 0] = stretch(yuv[:, :, 0])
result_yuv = np.clip(color.yuv2rgb(yuv), 0, 1)

# Method 4: YUV — 拉伸 Y，放大 UV 對比以增強色彩
yuv2 = color.rgb2yuv(img)
yuv2[:, :, 0] = stretch(yuv2[:, :, 0])
yuv2[:, :, 1] = yuv2[:, :, 1] * 1.5
yuv2[:, :, 2] = yuv2[:, :, 2] * 1.5
result_yuv_boost = np.clip(color.yuv2rgb(yuv2), 0, 1)

# Method 5: HSV — V 向均值壓縮（縮小 V 的分散程度，把亮暗往中間靠）
hsv5 = color.rgb2hsv(img)
v = stretch(hsv5[:, :, 2])
v_mean = v.mean()
hsv5[:, :, 2] = np.clip(v_mean + (v - v_mean) * 0.6, 0, 1)  # 壓縮到 60%
result_hsv_compress = color.hsv2rgb(hsv5)

# Method 6: YUV — Y 用 log transform 壓縮（高光壓更多，暗部保留，類 HDR tone mapping）
yuv6 = color.rgb2yuv(img)
y = stretch(yuv6[:, :, 0])
yuv6[:, :, 0] = np.log1p(y * 3) / np.log1p(3)  # log(1+3x)/log(4)，壓縮高光
result_yuv_log = np.clip(color.yuv2rgb(yuv6), 0, 1)

# Method 7: YUV — Y 用 sigmoid 壓縮（兩端都壓，分布集中在中段）
yuv7 = color.rgb2yuv(img)
y = stretch(yuv7[:, :, 0])
y_centered = y - 0.5
yuv7[:, :, 0] = 1 / (1 + np.exp(-y_centered * 4)) * 0.8 + 0.1  # sigmoid，範圍約 [0.1, 0.9]
result_yuv_sigmoid = np.clip(color.yuv2rgb(yuv7), 0, 1)


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
titles = [
    '原圖（白平衡後）',
    'HSV: V 拉伸',
    'HSV: V 拉伸 + S×1.3',
    'YUV: Y 拉伸',
    'YUV: Y 拉伸 + UV×1.5',
    'HSV: V 向均值壓縮×0.6',
    'YUV: Y log transform（壓高光）',
    'YUV: Y sigmoid（兩端壓縮）',
]
results = [img, result_hsv, result_hsv_sat, result_yuv, result_yuv_boost,
           result_hsv_compress, result_yuv_log, result_yuv_sigmoid]

for ax, title, res in zip(axes, titles, results):
    ax.imshow(res)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
axes[-1].axis('off')  # 最後一格空白

plt.tight_layout()
plt.savefig('test_color_compare.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("已儲存 test_color_compare.jpg")
