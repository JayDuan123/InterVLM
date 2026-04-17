"""
umap_overlay.py
8个标签各一种颜色，叠在同一张 UMAP 上
每个样本可以同时被多个标签涂色（后画的覆盖前画的）
画两张：h 空间 + z 空间
"""

import os, torch
import numpy as np
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

print("Loading data...")
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")
tr_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
te_lat = torch.load(f"{OUT_DIR}/test_latents.pt")

h_all = torch.cat([tr_emb['embeddings'], te_emb['embeddings']]).float().numpy()
z_all = torch.cat([tr_lat['z'], te_lat['z']]).numpy()

def get_label(key, na_val=-1):
    n_tr = len(tr_emb['embeddings'])
    n_te = len(te_emb['embeddings'])
    tr_l = tr_emb[key].numpy() if key in tr_emb else np.full(n_tr, na_val)
    te_l = te_emb[key].numpy() if key in te_emb else np.full(n_te, na_val)
    return np.concatenate([tr_l, te_l])

# 8个标签，每个一种颜色，只要有有效标签就画
LABEL_LAYERS = [
    ('hate',         -1,  '#E24B4A', 'hate',         lambda x: x >= 0),
    ('offensive',    -1,  '#F5A623', 'offensive',     lambda x: x >= 0),
    ('humor',        -1,  '#9B59B6', 'humor',         lambda x: x >= 0),
    ('stance',       -1,  '#2ECC71', 'stance',        lambda x: x >= 0),
    ('sentiment',   -99,  '#3498DB', 'sentiment',     lambda x: x != -99),
    ('sarcasm',      -1,  '#E67E22', 'sarcasm',       lambda x: x >= 0),
    ('motivational', -1,  '#1ABC9C', 'motivational',  lambda x: x >= 0),
    ('target',       -1,  '#E91E8C', 'target',        lambda x: x >= 0),
]

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

hate_arr = get_label('hate')
bias_all = (z_all[hate_arr==1]>0).mean(0) - (z_all[hate_arr==0]>0).mean(0)
top200   = np.argsort(np.abs(bias_all))[::-1][:200]
z_sub    = normalize(z_all[:, top200], norm='l2')
z_2d     = get_umap(z_sub, f"{OUT_DIR}/umap_z_coords.npy",
                    metric='cosine', n_neighbors=30, min_dist=0.2)

def plot_overlay(coords_2d, title, out_path):
    clean  = remove_outliers(coords_2d)
    coords = coords_2d[clean]

    fig, ax = plt.subplots(figsize=(13, 9))

    # 灰色背景
    ax.scatter(coords[:, 0], coords[:, 1],
               c='#EFEFEF', s=3, alpha=.12, linewidths=0, zorder=1)

    handles = []
    for (lkey, na_val, color, label, valid_fn) in LABEL_LAYERS:
        ldata = get_label(lkey, na_val)[clean]
        mask  = valid_fn(ldata)
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=5, alpha=.45,
                   linewidths=0, zorder=3)
        handles.append(mpatches.Patch(
            color=color,
            label=f"{label}  (n={mask.sum():,})"
        ))

    ax.legend(handles=handles, fontsize=9, loc='upper left',
              framealpha=.88, handlelength=1.2,
              borderpad=0.5, title='Label', title_fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP-1', fontsize=10)
    ax.set_ylabel('UMAP-2', fontsize=10)
    ax.grid(True, alpha=.07)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")

print("\nPlotting...")
plot_overlay(h_2d,
    'UMAP: CLIP Fused Representation (h)\n8 label dimensions overlaid (one color per label)',
    f"{FIG_DIR}/figE_umap_h_overlay.png")

plot_overlay(z_2d,
    'UMAP: SAE Latent Space (z)\n8 label dimensions overlaid (one color per label)',
    f"{FIG_DIR}/figF_umap_z_overlay.png")

print("\n完成！")
