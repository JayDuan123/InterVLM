"""
umap_density.py  —  UMAP with KDE density contours
对 h 和 z 空间各生成一张图：
  - 散点（半透明小点）
  - 每类的 KDE 密度等高线
  - 右下角标注 hate/benign 分离程度（silhouette score）
"""

import os, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

try:
    import umap as umap_lib
except ImportError:
    raise ImportError("pip install umap-learn")

OUT_DIR = "/projectnb/cepinet/users/Jay/InterVLP/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading data...")
train_lat  = torch.load(f"{OUT_DIR}/train_latents.pt")
test_lat   = torch.load(f"{OUT_DIR}/test_latents.pt")
norm_stats = torch.load(f"{OUT_DIR}/norm_stats.pt")

h_all   = torch.cat([train_lat['embeddings'], test_lat['embeddings']], dim=0).float().numpy()
z_all   = torch.cat([train_lat['z'],          test_lat['z']         ], dim=0).numpy()
lbl_all = torch.cat([train_lat['labels'],      test_lat['labels']    ]).numpy()

mean_h = norm_stats['mean'].numpy()
std_h  = norm_stats['std'].numpy()
h_norm = (h_all - mean_h) / (std_h + 1e-6)

# ── UMAP ─────────────────────────────────────────────────────
def get_umap(data, cache_path, metric='euclidean'):
    if os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return np.load(cache_path)
    print(f"  Running UMAP (metric={metric})...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42,
                             n_neighbors=15, min_dist=0.1,
                             metric=metric, verbose=False)
    coords = reducer.fit_transform(data)
    np.save(cache_path, coords)
    return coords

h_2d = get_umap(h_norm, f"{OUT_DIR}/umap_h_coords.npy", metric='euclidean')

# z: top-200 active features
hate_rate_all = (z_all[lbl_all==1] > 0).mean(axis=0)
ben_rate_all  = (z_all[lbl_all==0] > 0).mean(axis=0)
bias_all      = hate_rate_all - ben_rate_all
top200        = np.argsort(np.abs(bias_all))[::-1][:200]
z_sub    = normalize(z_all[:, top200], norm='l2')
z_2d     = get_umap(z_sub, f"{OUT_DIR}/umap_z_coords.npy", metric='cosine')

# ── 绘图函数 ──────────────────────────────────────────────────
def plot_umap_density(coords, labels, title, out_path):
    hate_mask   = labels == 1
    benign_mask = labels == 0

    # silhouette score（-1~1，越高越分离）
    try:
        sil = silhouette_score(coords, labels, sample_size=2000, random_state=42)
    except Exception:
        sil = float('nan')

    fig, ax = plt.subplots(figsize=(9, 7))

    # 散点（小、半透明）
    ax.scatter(coords[benign_mask, 0], coords[benign_mask, 1],
               c='#378ADD', s=5, alpha=.25, linewidths=0, label='Benign', zorder=2)
    ax.scatter(coords[hate_mask, 0],   coords[hate_mask, 1],
               c='#E24B4A', s=5, alpha=.25, linewidths=0, label='Hate',   zorder=2)

    # KDE 等高线
    x_min, x_max = coords[:, 0].min() - .3, coords[:, 0].max() + .3
    y_min, y_max = coords[:, 1].min() - .3, coords[:, 1].max() + .3
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid   = np.vstack([xx.ravel(), yy.ravel()])

    for mask, color, cmap_name in [
        (benign_mask, '#1a6bb5', 'Blues'),
        (hate_mask,   '#b52020', 'Reds'),
    ]:
        pts = coords[mask].T
        if pts.shape[1] < 10:
            continue
        kde = gaussian_kde(pts, bw_method=0.25)
        z   = kde(grid).reshape(xx.shape)
        # 归一化到 [0,1]
        z   = (z - z.min()) / (z.max() - z.min() + 1e-10)
        ax.contour(xx, yy, z,
                   levels=[0.15, 0.35, 0.60, 0.85],
                   colors=color, linewidths=[.6, .9, 1.2, 1.5],
                   alpha=.7, zorder=3)

    # 统计标注
    n_hate   = hate_mask.sum()
    n_benign = benign_mask.sum()
    info = (f"n={len(labels)}  hate={n_hate}  benign={n_benign}\n"
            f"Silhouette score = {sil:.3f}")
    ax.text(.98, .02, info, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8.5,
            color='#444',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd', alpha=.85))

    ax.legend(fontsize=9, markerscale=3, framealpha=.85)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.grid(True, alpha=.08)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}  (silhouette={sil:.3f})")
    return sil

# ── 生成两张图 ────────────────────────────────────────────────
print("\nPlotting UMAP with density contours...")

sil_h = plot_umap_density(
    h_2d, lbl_all,
    title='UMAP: MemeCLIP Fused Representation (h)\n'
          'contours = KDE density per class',
    out_path=f"{FIG_DIR}/fig3_umap_h_density.png"
)

sil_z = plot_umap_density(
    z_2d, lbl_all,
    title='UMAP: SAE Latent Space (z, top-200 features)\n'
          'contours = KDE density per class',
    out_path=f"{FIG_DIR}/fig4_umap_z_density.png"
)

print(f"\n=== Silhouette 对比 ===")
print(f"  h 空间: {sil_h:.4f}")
print(f"  z 空间: {sil_z:.4f}")
if sil_z > sil_h:
    print(f"  → SAE latent space 分离度更高 (+{sil_z-sil_h:.4f})")
else:
    print(f"  → h 空间分离度更高，SAE 更侧重 concept 分解而非分类")
print("\n完成！")
