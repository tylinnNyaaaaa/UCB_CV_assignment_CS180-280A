import numpy as np
from skimage import io
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

print("B shape:", b.shape)
print("G shape:", g.shape)
print("R shape:", r.shape)

color_img = np.stack([r, g, b], axis=2)
plt.imsave('output_raw_tif.jpg', color_img)
print("已儲存 output_raw_tif.jpg")