import numpy as np
from scipy.ndimage import map_coordinates


def computeH(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """
    Compute homography H such that p' ~ H @ p (homogeneous coords).

    Args:
        im1_pts: (n, 2) source (x, y) points, n >= 4
        im2_pts: (n, 2) destination (x, y) points

    Returns:
        H: (3, 3) with H[2,2] = 1
    """
    n = im1_pts.shape[0]
    x  = im1_pts[:, 0]
    y  = im1_pts[:, 1]
    xp = im2_pts[:, 0]
    yp = im2_pts[:, 1]

    zeros = np.zeros(n)
    ones  = np.ones(n)

    rows_top = np.stack([x, y, ones, zeros, zeros, zeros, -x * xp, -y * xp], axis=1)
    rows_bot = np.stack([zeros, zeros, zeros, x, y, ones, -x * yp, -y * yp], axis=1)

    A = np.empty((2 * n, 8), dtype=np.float64)
    A[0::2] = rows_top
    A[1::2] = rows_bot

    b = np.empty(2 * n, dtype=np.float64)
    b[0::2] = xp
    b[1::2] = yp

    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ])


def warpImage(im: np.ndarray, H: np.ndarray) -> tuple:
    """
    Inverse-warp `im` through homography H.

    Args:
        im: (H, W, C) float64 image in [0, 1]
        H:  (3, 3) homography mapping im -> destination

    Returns:
        warped: (H_out, W_out, C) float64
        mask:   (H_out, W_out)   bool, True where source coords are in-bounds
    """
    h_im, w_im, C = im.shape

    # --- Step 1: forward-map corners to find output bounding box ---
    corners = np.array([[0,    0,    1],
                        [w_im, 0,    1],
                        [0,    h_im, 1],
                        [w_im, h_im, 1]], dtype=np.float64).T   # (3, 4)
    proj = H @ corners
    proj /= proj[2:3]

    x_min, x_max = proj[0].min(), proj[0].max()
    y_min, y_max = proj[1].min(), proj[1].max()

    W_out = int(np.ceil(x_max - x_min))
    H_out = int(np.ceil(y_max - y_min))

    # --- Step 2: destination pixel grid, shifted by bounding-box offset ---
    x_dst = np.arange(W_out, dtype=np.float64) + x_min
    y_dst = np.arange(H_out, dtype=np.float64) + y_min
    xs, ys = np.meshgrid(x_dst, y_dst)                          # (H_out, W_out)
    dst_hom = np.stack([xs, ys, np.ones_like(xs)], axis=0).reshape(3, -1)  # (3, N)

    # --- Step 3: inverse map to source ---
    H_inv = np.linalg.inv(H)
    src = H_inv @ dst_hom
    src /= src[2:3]
    x_src = src[0]   # (N,)
    y_src = src[1]   # (N,)

    # --- Step 4: interpolate, vectorized over channels ---
    valid = (x_src >= 0) & (x_src < w_im) & (y_src >= 0) & (y_src < h_im)
    mask  = valid.reshape(H_out, W_out)

    N = H_out * W_out
    # Tile spatial coords C times; repeat channel index N times each
    rows_all  = np.tile(y_src, C)               # (C*N,)  row = y
    cols_all  = np.tile(x_src, C)               # (C*N,)  col = x
    chans_all = np.repeat(np.arange(C), N)      # (C*N,)

    sampled = map_coordinates(im, [rows_all, cols_all, chans_all],
                              order=1, mode='constant', cval=0.0)
    warped = sampled.reshape(C, H_out, W_out).transpose(1, 2, 0)

    return warped, mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.image import imread as mpl_imread

    # --- UTF-8 terminal output ---
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    # --- 中文字型：依序嘗試 Windows 內建繁/簡中文字型 ---
    _zh_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'PMingLiU',
                 'MingLiU', 'DFKai-SB', 'SimHei']
    _available = {f.name for f in fm.fontManager.ttflist}
    _chosen = next((f for f in _zh_fonts if f in _available), None)
    if _chosen:
        plt.rcParams['font.family'] = [_chosen, 'DejaVu Sans']
    else:
        print("  [警告] 找不到中文字型，圖片標題將顯示為方框")

    # --- Test 1: computeH round-trip ---r
    print("Test 1: computeH round-trip")
    H_true = np.array([
        [1.2,   0.3,  50.0],
        [0.1,   0.9,  30.0],
        [0.001, 5e-4,  1.0],
    ])
    rng = np.random.default_rng(42)
    src = rng.uniform([0, 0], [640, 480], size=(8, 2))
    src_h = np.hstack([src, np.ones((8, 1))])
    dst_h = (H_true @ src_h.T).T
    dst   = dst_h[:, :2] / dst_h[:, 2:3]
    H_est = computeH(src, dst)
    err1  = np.max(np.abs(H_est - H_true))        # H_true[2,2] already == 1
    print(f"  max |H_est - H_true| = {err1:.2e}")
    assert err1 < 1e-5, f"FAILED: error {err1}"
    print("  PASSED\n")

    # --- Test 2: warpImage identity ---
    print("Test 2: warpImage identity")
    rng2   = np.random.default_rng(0)
    im_rnd = rng2.random((100, 100, 3))
    warped_id, mask_id = warpImage(im_rnd, np.eye(3))
    err2 = np.max(np.abs(warped_id - im_rnd))
    print(f"  max |warped - original| = {err2:.2e},  mask all True: {mask_id.all()}")
    assert err2 < 1e-4,    f"FAILED: error {err2}"
    assert mask_id.all(),   "FAILED: mask not all True"
    print("  PASSED\n")

    # --- Test 3: visual sanity check ---
    print("Test 3: warpImage visual sanity check")
    img_path = 'img1.jpg'
    if os.path.exists(img_path):
        im_real = mpl_imread(img_path).astype(np.float64) / 255.0
    else:
        # checkerboard fallback
        sz, tile = 300, 30
        grid = (np.arange(sz)[:, None] // tile + np.arange(sz)[None, :] // tile) % 2
        im_real = np.stack([grid] * 3, axis=-1).astype(np.float64)

    h_r, w_r = im_real.shape[:2]
    src_c = np.array([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]], dtype=np.float64)

    def _H(dst):
        return computeH(src_c, np.array(dst, dtype=np.float64))

    # (英文名, 中文名, 中文說明, H)
    warps = [
        ("top pinch",   "上方收縮", "頂部兩角向內縮 10%",
         _H([[w_r*0.10, h_r*0.05], [w_r*0.90, h_r*0.05],
              [w_r,      h_r      ], [0,        h_r     ]])),
        ("bottom pinch","下方收縮", "底部兩角向內縮 10%",
         _H([[0,         0        ], [w_r,      0       ],
              [w_r*0.90, h_r*0.95 ], [w_r*0.10, h_r*0.95]])),
        ("right pinch", "右方收縮", "右側兩角向內縮 5%",
         _H([[0,         0        ], [w_r*0.95, h_r*0.10],
              [w_r*0.95, h_r*0.90 ], [0,        h_r     ]])),
        ("left pinch",  "左方收縮", "左側兩角向內縮 5%",
         _H([[w_r*0.05, h_r*0.10 ], [w_r,      0       ],
              [w_r,      h_r       ], [w_r*0.05, h_r*0.90]])),
    ]

    # --- 計算所有 warp 並收集結果 ---
    results = []
    for eng, chi, desc, H_w in warps:
        warped, mask = warpImage(im_real, H_w)
        coverage = mask.mean() * 100
        results.append((eng, chi, desc, warped, coverage))

    # --- 中文結果表格 ---
    sep   = "─" * 54
    hsep  = "═" * 54
    print(f"\n  {hsep}")
    print(f"  {'變形類型':<6}  {'輸出解析度':<14}  {'有效覆蓋率':<10}  說明")
    print(f"  {sep}")
    for _, chi, desc, warped, cov in results:
        size = f"{warped.shape[1]}×{warped.shape[0]}"
        print(f"  {chi:<6}  {size:<14}  {cov:>6.1f}%    {desc}")
    print(f"  {hsep}\n")

    # --- 圖表排版：單行，原圖在最左 ---
    n_panels = 1 + len(warps)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5 * n_panels, 5),
                             gridspec_kw={'wspace': 0.05})
    fig.suptitle('warpImage — 透視變形比較', fontsize=13, fontweight='bold', y=1.01)

    axes[0].imshow(np.clip(im_real, 0, 1))
    axes[0].set_title(f'原圖\n{w_r}×{h_r}', fontsize=10, pad=5)
    axes[0].axis('off')

    for ax, (_, chi, desc, warped, cov) in zip(axes[1:], results):
        ax.imshow(np.clip(warped, 0, 1))
        ax.set_title(f'{chi}（{desc}）\n'
                     f'{warped.shape[1]}×{warped.shape[0]}  覆蓋 {cov:.0f}%',
                     fontsize=10, pad=5)
        ax.axis('off')

    plt.savefig('test_warp_output.png', dpi=150, bbox_inches='tight')
    print("  Saved test_warp_output.png")
    print("  PASSED\n")

    print("All tests passed.")