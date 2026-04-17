"""
visualize_v2.py
从 concepts_v2.json 读取标注，生成所有图
fig1: feature scatter
fig2: polarity bar chart
fig3/4: UMAP with density
fig6: decoder UMAP annotated
fig7: three-group bar chart
"""

import os, json, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

try:
    import umap as umap_lib
except ImportError:
    raise ImportError("pip install umap-learn")

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载 concept 数据 ─────────────────────────────────────────
with open(f"{OUT_DIR}/concepts_v2.json") as f:
    CONCEPTS = json.load(f)

df_c = pd.DataFrame(CONCEPTS)
TYPE_COLOR = {'hate':'#E24B4A','mild-hate':'#EF9F27','neutral':'#888780','benign':'#378ADD'}
TYPE_LABEL = {'hate':'Hate-biased','mild-hate':'Mild hate','neutral':'Neutral','benign':'Benign-biased'}

# ── 加载 embedding / latent ───────────────────────────────────
print("Loading data...")
train_emb  = torch.load(f"{OUT_DIR}/train_embeddings.pt")
test_emb   = torch.load(f"{OUT_DIR}/test_embeddings.pt")
train_lat  = torch.load(f"{OUT_DIR}/train_latents.pt")
test_lat   = torch.load(f"{OUT_DIR}/test_latents.pt")

h_all   = torch.cat([train_emb['embeddings'], test_emb['embeddings']]).float().numpy()
z_all   = torch.cat([train_lat['z'],          test_lat['z']         ]).numpy()
lbl_all = torch.cat([train_emb['labels'],      test_emb['labels']    ]).numpy()

# norm stats
mean_h = h_all.mean(axis=0); std_h = h_all.std(axis=0).clip(1e-6)
h_norm = (h_all - mean_h) / std_h

# ═══════════════════════════════════════════════════════════════
# FIG 1: Feature scatter
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
for _, row in df_c.iterrows():
    color = TYPE_COLOR[row['type']]
    size  = 80 + (row['rate'] - df_c['rate'].min()) / (df_c['rate'].max() - df_c['rate'].min()) * 400
    ax.scatter(row['bias'], row['rate'], s=size, color=color, alpha=.75,
               edgecolors=color, linewidths=.8, zorder=3)
    ax.annotate(f"#{row['id']}", (row['bias'], row['rate']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0,5), textcoords='offset points', color='#444')

ax.axvline(0, color='gray', lw=.8, ls='--', alpha=.5)
ax.set_xlabel('Hate Bias', fontsize=11)
ax.set_ylabel('Activation Rate', fontsize=11)
ax.set_title('InterVLM: SAE Feature Map (CLIP ViT-L/14)\n(size = activation rate)', fontsize=12)
ax.legend(handles=[mpatches.Patch(color=TYPE_COLOR[t], label=TYPE_LABEL[t])
                   for t in ['hate','mild-hate','neutral','benign']],
          loc='upper left', fontsize=9)
ax.grid(True, alpha=.15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_feature_scatter.png", dpi=150)
plt.close(); print("✓ fig1")

# ═══════════════════════════════════════════════════════════════
# FIG 2: Polarity bar chart (top 8 each side)
# ═══════════════════════════════════════════════════════════════
top_hate   = df_c.nlargest(8,'bias').sort_values('bias')
top_benign = df_c.nsmallest(8,'bias').sort_values('bias',ascending=False)
fig, axes  = plt.subplots(1,2,figsize=(15,5))
for ax, sub, title, color in [
    (axes[0], top_hate,   'Top Hate-Biased Features',   '#E24B4A'),
    (axes[1], top_benign, 'Top Benign-Biased Features', '#378ADD'),
]:
    labels = [f"#{int(r.id)} {r.concept[:30]}" for _,r in sub.iterrows()]
    bars = ax.barh(labels, sub['bias'].abs(), color=color, alpha=.75, edgecolor='none')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_xlabel('|Hate Bias|', fontsize=10); ax.set_title(title, fontsize=11)
    ax.tick_params(axis='y', labelsize=8); ax.grid(axis='x', alpha=.2)
plt.suptitle('InterVLM: Feature Polarization (CLIP ViT-L/14)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_feature_polarity.png", dpi=150, bbox_inches='tight')
plt.close(); print("✓ fig2")

# ═══════════════════════════════════════════════════════════════
# FIG 3 & 4: UMAP with KDE density
# ═══════════════════════════════════════════════════════════════
def get_umap(data, cache, metric='euclidean', n_neighbors=15, min_dist=0.1, force=False):
    if os.path.exists(cache) and not force:
        return np.load(cache)
    print(f"  Running UMAP ({metric}, nn={n_neighbors}, md={min_dist})...")
    r = umap_lib.UMAP(n_components=2, random_state=42,
                       n_neighbors=n_neighbors, min_dist=min_dist,
                       metric=metric, verbose=False)
    coords = r.fit_transform(data)
    np.save(cache, coords)
    return coords

def remove_outliers(coords, labels):
    """移除 IQR 3倍以外的 outlier"""
    mask = np.ones(len(coords), dtype=bool)
    for dim in range(2):
        q1, q3 = np.percentile(coords[:, dim], [5, 95])
        pad = (q3 - q1) * 1.5
        mask &= (coords[:, dim] >= q1 - pad) & (coords[:, dim] <= q3 + pad)
    return coords[mask], labels[mask]

def plot_density(coords, labels, title, out_path):
    # 过滤 outlier
    coords, labels = remove_outliers(coords, labels)
    hate_m  = labels==1; ben_m = labels==0
    try: sil = silhouette_score(coords, labels, sample_size=min(2000,len(labels)), random_state=42)
    except: sil = float('nan')
    fig, ax = plt.subplots(figsize=(9,7))
    ax.scatter(coords[ben_m,0],  coords[ben_m,1],  c='#378ADD', s=5, alpha=.25, linewidths=0, label='Benign')
    ax.scatter(coords[hate_m,0], coords[hate_m,1], c='#E24B4A', s=5, alpha=.25, linewidths=0, label='Hate')
    x0,x1 = coords[:,0].min()-.3, coords[:,0].max()+.3
    y0,y1 = coords[:,1].min()-.3, coords[:,1].max()+.3
    xx,yy = np.mgrid[x0:x1:200j, y0:y1:200j]
    grid  = np.vstack([xx.ravel(), yy.ravel()])
    for mask, color in [(ben_m,'#1a6bb5'),(hate_m,'#b52020')]:
        pts = coords[mask].T
        if pts.shape[1]<10: continue
        kde = gaussian_kde(pts, bw_method=0.3)
        z   = kde(grid).reshape(xx.shape)
        z   = (z-z.min())/(z.max()-z.min()+1e-10)
        ax.contour(xx,yy,z,levels=[.15,.35,.60,.85],
                   colors=color, linewidths=[.6,.9,1.2,1.5], alpha=.7, zorder=3)
    ax.text(.98,.02, f"n={len(labels)}  hate={hate_m.sum()}  benign={ben_m.sum()}\nSilhouette={sil:.3f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd', alpha=.85))
    ax.legend(fontsize=9, markerscale=3, framealpha=.85)
    ax.set_title(title, fontsize=11); ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.grid(True, alpha=.08); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    return sil

print("Running UMAPs...")
h_2d = get_umap(h_norm, f"{OUT_DIR}/umap_h_coords.npy",
                n_neighbors=15, min_dist=0.1)

bias_all = (z_all[lbl_all==1]>0).mean(0) - (z_all[lbl_all==0]>0).mean(0)
top200   = np.argsort(np.abs(bias_all))[::-1][:200]
z_sub    = normalize(z_all[:,top200], norm='l2')
z_2d     = get_umap(z_sub, f"{OUT_DIR}/umap_z_coords.npy",
                    metric='cosine', n_neighbors=30, min_dist=0.2, force=True)

sil_h = plot_density(h_2d, lbl_all,
    'UMAP: CLIP Fused Representation (h)\ncontours = KDE density per class',
    f"{FIG_DIR}/fig3_umap_h_density.png")
sil_z = plot_density(z_2d, lbl_all,
    'UMAP: SAE Latent Space (z, top-200 bias features)\ncontours = KDE density per class',
    f"{FIG_DIR}/fig4_umap_z_density.png")
print(f"✓ fig3 (sil={sil_h:.3f})  fig4 (sil={sil_z:.3f})")

# ═══════════════════════════════════════════════════════════════
# FIG 5: UMAP colored by dominant SAE feature (concept clusters)
# ═══════════════════════════════════════════════════════════════
print("Plotting fig5: concept cluster UMAP...")

z_full = np.concatenate([train_lat['z'].numpy(), test_lat['z'].numpy()])
z_top  = z_full[:, top200]
dom_local  = z_top.argmax(axis=1)
dom_global = top200[dom_local]

dom_counts = pd.Series(dom_global).value_counts()
top20_dom  = dom_counts.head(100).index.tolist()

# outlier mask（与 remove_outliers 一致）
def get_clean_mask(coords):
    mask = np.ones(len(coords), dtype=bool)
    for dim in range(2):
        q1, q3 = np.percentile(coords[:, dim], [5, 95])
        pad = (q3 - q1) * 1.5
        mask &= (coords[:, dim] >= q1 - pad) & (coords[:, dim] <= q3 + pad)
    return mask

clean_mask  = get_clean_mask(z_2d)
z_2d_clean  = z_2d[clean_mask]
dom_clean   = dom_global[clean_mask]
lbl_clean   = lbl_all[clean_mask]

# 读取每个 feature 的 hate_bias
concept_report = pd.read_csv(f"{OUT_DIR}/concept_report.csv")
fid2bias = dict(zip(concept_report['feature_id'].astype(int),
                    concept_report['hate_bias']))
concept_lookup = {c['id']: c['concept'] for c in CONCEPTS}

fig, ax = plt.subplots(figsize=(12, 8))

# 灰色背景：不在 top20 的点
bg = np.array([f not in set(top20_dom) for f in dom_clean])
ax.scatter(z_2d_clean[bg,0], z_2d_clean[bg,1],
           c='#dddddd', s=3, alpha=.15, linewidths=0, zorder=1)

# top-20 的点：颜色按 hate_bias，用 RdBu_r colormap
fg_mask  = ~bg
fg_coords = z_2d_clean[fg_mask]
fg_feats  = dom_clean[fg_mask]
fg_bias   = np.array([fid2bias.get(int(f), 0.0) for f in fg_feats])

# 归一化到 [-max_abs, +max_abs] 对称
max_abs = np.abs(fg_bias).max()
sc = ax.scatter(fg_coords[:,0], fg_coords[:,1],
                c=fg_bias, cmap='RdBu_r',
                vmin=-max_abs, vmax=max_abs,
                s=22, alpha=.85, linewidths=0, zorder=3)
plt.colorbar(sc, ax=ax, label='Hate Bias', shrink=.7, pad=.01)

# 标注每个 cluster 的 feature id + concept
for fid in top20_dom:
    pts = fg_coords[fg_feats == fid]
    if len(pts) == 0: continue
    cx, cy = pts.mean(axis=0)
    name   = concept_lookup.get(int(fid), f'#{fid}')
    bias   = fid2bias.get(int(fid), 0.0)
    ax.annotate(
        f"#{fid}\n{name[:20]}",
        (cx, cy), fontsize=6.5, ha='center', va='bottom',
        xytext=(0, 6), textcoords='offset points',
        color='#222',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#bbb', alpha=.75),
        zorder=5
    )

ax.set_title('InterVLM: SAE Concept Clusters (top-20 dominant features)\n'
             'color = hate bias per feature  (red = hate, blue = benign)', fontsize=11)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.grid(True, alpha=.08)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_concept_clusters.png", dpi=150)
plt.close()
print("✓ fig5_concept_clusters.png")

# ═══════════════════════════════════════════════════════════════
# FIG 6: Decoder UMAP annotated with ellipses
# ═══════════════════════════════════════════════════════════════
ckpt     = torch.load(f"{OUT_DIR}/sae_model.pt", map_location='cpu')
decoder_w = ckpt['state_dict']['decoder.weight'].numpy()
W        = normalize(decoder_w.T, norm='l2')
D        = W.shape[0]

W_2d = get_umap(W, f"{OUT_DIR}/decoder_umap_coords.npy", metric='cosine')

ann_ids   = [c['id'] for c in CONCEPTS if c['id']<D]
ann_x     = W_2d[[c['id'] for c in CONCEPTS if c['id']<D], 0]
ann_y     = W_2d[[c['id'] for c in CONCEPTS if c['id']<D], 1]
ann_bias  = np.array([c['bias'] for c in CONCEPTS if c['id']<D])
ann_rate  = np.array([c['rate'] for c in CONCEPTS if c['id']<D])
ann_names = [c['concept'] for c in CONCEPTS if c['id']<D]
ann_types = [c['type']    for c in CONCEPTS if c['id']<D]

def fit_ellipse(xs, ys, pad=0.25):
    return xs.mean(), ys.mean(), (xs.max()-xs.min())/2+pad, (ys.max()-ys.min())/2+pad

fig, ax = plt.subplots(figsize=(13,9))
ax.scatter(W_2d[:,0], W_2d[:,1], s=2, alpha=.1, color='#aaaaaa', linewidths=0, zorder=1)

for i, (fid, name) in enumerate(zip(ann_ids, ann_names)):
    c = TYPE_COLOR[ann_types[i]]
    ax.scatter(ann_x[i], ann_y[i],
               s=60+ann_rate[i]/ann_rate.max()*280, color=c,
               edgecolors='white', linewidths=.8, alpha=.92, zorder=4)
    ax.annotate(f"#{fid} {name}", (ann_x[i], ann_y[i]),
                fontsize=7.2, xytext=(5,4), textcoords='offset points',
                color='#222', bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#ccc', alpha=.78), zorder=5)

# ellipses
for gtype, color, label in [('hate','#E24B4A','Hate cluster'),('benign','#378ADD','Benign cluster')]:
    mask = np.array([t in (gtype,'mild-hate' if gtype=='hate' else gtype) for t in ann_types])
    if mask.sum()>=2:
        cx,cy,rx,ry = fit_ellipse(ann_x[mask], ann_y[mask])
        ax.add_patch(Ellipse((cx,cy), rx*2, ry*2,
                             edgecolor=color, facecolor='none',
                             linestyle='--', linewidth=1.6, alpha=.8, zorder=3))
        offset = rx-0.05 if gtype=='hate' else -rx+0.05
        ha = 'right' if gtype=='hate' else 'left'
        ax.text(cx+offset, cy+ry+0.08, label, color=color, fontsize=8.5, fontweight='bold', ha=ha)

ax.legend(handles=[mpatches.Patch(color=TYPE_COLOR[t], label=TYPE_LABEL[t])
                   for t in ['hate','mild-hate','neutral','benign']],
          loc='lower right', fontsize=8.5, framealpha=.85)
ax.set_title('InterVLM: Annotated SAE Features in Decoder Space (CLIP ViT-L/14)\n'
             '(color=type, size=activation rate, dashed=clusters)', fontsize=11)
ax.set_xlabel('UMAP-1 (cosine)'); ax.set_ylabel('UMAP-2')
ax.grid(True, alpha=.1); plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_decoder_annotated_v2.png", dpi=150)
plt.close(); print("✓ fig6")

# ═══════════════════════════════════════════════════════════════
# FIG 7: Three-group polarity bar chart
# ═══════════════════════════════════════════════════════════════
df_ann = pd.DataFrame({
    'id':      ann_ids,
    'concept': ann_names,
    'bias':    ann_bias,
    'rate':    ann_rate,
    'group':   ['hate' if b>0.015 else 'benign' if b<-0.015 else 'neutral' for b in ann_bias],
})

fig, axes = plt.subplots(1,3,figsize=(16,7))
for ax, g, title, color in [
    (axes[0],'hate',   'Hate-biased\n(bias > +0.015)',   '#E24B4A'),
    (axes[1],'neutral','Neutral\n(|bias| ≤ 0.015)',      '#888780'),
    (axes[2],'benign', 'Benign-biased\n(bias < −0.015)', '#378ADD'),
]:
    sub = df_ann[df_ann.group==g].sort_values('bias', ascending=(g!='hate'))
    if len(sub)==0: ax.set_visible(False); continue
    labels = [f"#{int(r.id)}  {r.concept[:28]}" for _,r in sub.iterrows()]
    bars   = ax.barh(labels, sub['bias'].values, color=color, alpha=.75, height=.6, edgecolor='none')
    ax.bar_label(bars, fmt='%+.3f', padding=3, fontsize=7.5, color='#333')
    for j,(v,s) in enumerate(zip(sub['bias'].values, sub['rate'].values)):
        ax.scatter(v+(0.002 if v>=0 else -0.002), j,
                   s=40+s/ann_rate.max()*120, color=color,
                   edgecolors='white', linewidths=.6, alpha=.9, zorder=5)
    ax.axvline(0, color='gray', lw=.7, ls='--', alpha=.6)
    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    ax.set_xlabel('Hate Bias', fontsize=9)
    ax.tick_params(axis='y', labelsize=7.5); ax.grid(axis='x', alpha=.15)

plt.suptitle('InterVLM: SAE Concept Features Grouped by Polarity (CLIP ViT-L/14)\n'
             '(dot size = activation rate)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_concept_polarity_v2.png", dpi=150, bbox_inches='tight')
plt.close(); print("✓ fig7")

print(f"\n全部完成！图片: {FIG_DIR}")
print(f"Silhouette — h: {sil_h:.4f}  z: {sil_z:.4f}")
