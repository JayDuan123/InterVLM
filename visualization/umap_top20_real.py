"""
umap_top20_real.py
用真实数据画 top-20 dominant SAE feature 的两张 UMAP
h 空间：混在一起
z 空间：按 feature cluster 分开
"""

import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import normalize

try:
    import umap as umap_lib
except ImportError:
    raise ImportError("pip install umap-learn")

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading data...")
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")
tr_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
te_lat = torch.load(f"{OUT_DIR}/test_latents.pt")

h_all = torch.cat([tr_emb['embeddings'], te_emb['embeddings']]).float().numpy()
z_all = torch.cat([tr_lat['z'],          te_lat['z']         ]).numpy()
hate  = torch.cat([tr_emb['hate'],        te_emb['hate']      ]).numpy()

# ── 计算 top-20 dominant features ────────────────────────────
print("Computing top-20 dominant features...")
bias_all   = (z_all[hate==1]>0).mean(0) - (z_all[hate==0]>0).mean(0)
top200     = np.argsort(np.abs(bias_all))[::-1][:200]
z_sub      = normalize(z_all[:, top200], norm='l2')

act_rate   = (z_all > 0).mean(axis=0)
dom_feat   = z_all[:, top200].argmax(axis=1)   # index in top200
dom_feat_g = top200[dom_feat]                  # global feature id

dom_counts = pd.Series(dom_feat_g).value_counts()
top20_ids  = dom_counts.head(20).index.tolist()

print(f"  Top-20 feature ids: {top20_ids}")

# 颜色：根据 hate_bias 分配（红=hate，蓝=benign，绿/紫=中性）
def bias_to_color(b):
    if b > 0.02:   return plt.cm.Reds(0.5 + b * 8)
    if b < -0.02:  return plt.cm.Blues(0.5 + abs(b) * 8)
    return plt.cm.Greens(0.4)

feat_colors = {}
for fid in top20_ids:
    b = bias_all[fid]
    feat_colors[fid] = bias_to_color(b)

# ── UMAP 坐标 ─────────────────────────────────────────────────
def get_umap(data, cache, metric='euclidean', n_neighbors=15, min_dist=0.1):
    if os.path.exists(cache):
        print(f"  Cached: {os.path.basename(cache)}")
        return np.load(cache)
    print(f"  Running UMAP ({metric})...")
    r = umap_lib.UMAP(n_components=2, random_state=42,
                       n_neighbors=n_neighbors, min_dist=min_dist,
                       metric=metric, verbose=False)
    coords = r.fit_transform(data)
    np.save(cache, coords)
    return coords

def remove_outliers(coords):
    mask = np.ones(len(coords), dtype=bool)
    for dim in range(2):
        q1, q3 = np.percentile(coords[:, dim], [5, 95])
        pad = (q3 - q1) * 1.5
        mask &= (coords[:, dim] >= q1 - pad) & (coords[:, dim] <= q3 + pad)
    return mask

print("Computing UMAPs...")
mean_h = h_all.mean(0); std_h = h_all.std(0).clip(1e-6)
h_norm = (h_all - mean_h) / std_h
h_2d   = get_umap(h_norm, f"{OUT_DIR}/umap_h_coords.npy",
                  n_neighbors=15, min_dist=0.1)
z_2d   = get_umap(z_sub,  f"{OUT_DIR}/umap_z_coords.npy",
                  metric='cosine', n_neighbors=30, min_dist=0.2)

# ── 画图 ──────────────────────────────────────────────────────
def make_plot(coords_2d, dom_arr, title, out_path, mixed=False):
    clean  = remove_outliers(coords_2d)
    coords = coords_2d[clean]
    dom_c  = dom_arr[clean]

    fig, ax = plt.subplots(figsize=(13, 10))

    # 灰色背景（不在 top-20 的点）
    bg = np.array([f not in feat_colors for f in dom_c])
    ax.scatter(coords[bg, 0], coords[bg, 1],
               c='#EBEBEB', s=3, alpha=.15, linewidths=0, zorder=1)

    handles = []
    for fid in top20_ids:
        color = feat_colors[fid]
        mask  = dom_c == fid
        if mask.sum() == 0: continue

        if mixed:
            # h 空间：随机打散，不加扰动
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color], s=10, alpha=.55, linewidths=0, zorder=3)
        else:
            # z 空间：自然聚类
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color], s=14, alpha=.80, linewidths=0, zorder=3)
            # 标注 cluster 中心
            cx, cy = coords[mask].mean(axis=0)
            ax.annotate(
                f"#{fid}",
                (cx, cy), fontsize=7, ha='center', va='center',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=.8),
                zorder=5
            )

        b = bias_all[fid]
        handles.append(mpatches.Patch(
            color=color,
            label=f"#{fid}  bias={b:+.3f}  n={mask.sum()}"
        ))

    ax.legend(handles=handles, fontsize=7.5, loc='upper left',
              framealpha=.88, ncol=2, handlelength=1.2,
              title='Top-20 SAE Features', title_fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP-1', fontsize=11)
    ax.set_ylabel('UMAP-2', fontsize=11)
    ax.grid(True, alpha=.08)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")

print("\nPlotting...")
make_plot(h_2d, dom_feat_g,
    'UMAP: CLIP Fused Representation (h)\nTop-20 dominant SAE features — globally mixed',
    f"{FIG_DIR}/umap_h_top20.png",
    mixed=True)

make_plot(z_2d, dom_feat_g,
    'UMAP: SAE Latent Space (z)\nTop-20 dominant features — concept clusters',
    f"{FIG_DIR}/umap_z_top20.png",
    mixed=False)

print("\n完成！")
