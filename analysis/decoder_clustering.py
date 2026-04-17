"""
decoder_clustering.py  —  InterVLM Decoder Weight Vector Analysis
对 SAE decoder 的 4096 个 weight vector 做聚类 + UMAP
每个 vector d_i ∈ R^1024 代表一个 feature 在原始 embedding 空间的"方向"
相似的 vector → 语义上相近的 concept

输出:
  sae_outputs/figures/fig5_decoder_umap.png     全局 UMAP，按聚类上色
  sae_outputs/figures/fig6_decoder_umap_top.png 只展示高激活 feature，标注 concept 名
  sae_outputs/decoder_clusters.csv             每个 feature 的 cluster 归属
"""

import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError:
    raise ImportError("请先: pip install umap-learn")

OUT_DIR = "/projectnb/cepinet/users/Jay/InterVLP/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载 SAE decoder weights ──────────────────────────────────
print("Loading SAE decoder weights...")
ckpt = torch.load(f"{OUT_DIR}/sae_model.pt", map_location='cpu')

# decoder.weight shape: [input_dim, latent_dim] = [1024, 4096]
# 每一列 decoder.weight[:, i] 是 feature i 的 direction vector
decoder_w = ckpt['state_dict']['decoder.weight'].numpy()  # [1024, 4096]
D = decoder_w.shape[1]   # 4096 features
print(f"  decoder weight: {decoder_w.shape}  →  {D} feature vectors of dim {decoder_w.shape[0]}")

# 转置：每行是一个 feature 的 vector
W = decoder_w.T           # [4096, 1024]
W_norm = normalize(W, norm='l2')  # L2 归一化，方便余弦距离聚类

# ── 加载激活率统计 ────────────────────────────────────────────
print("Loading activation stats...")
train_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
test_lat  = torch.load(f"{OUT_DIR}/test_latents.pt")
z_all = np.concatenate([train_lat['z'].numpy(), test_lat['z'].numpy()], axis=0)
lbl_all = np.concatenate([train_lat['labels'].numpy(), test_lat['labels'].numpy()])

act_rate  = (z_all > 0).mean(axis=0)           # [4096]
hate_rate = (z_all[lbl_all==1] > 0).mean(axis=0)
ben_rate  = (z_all[lbl_all==0] > 0).mean(axis=0)
hate_bias = hate_rate - ben_rate

# ── KMeans 聚类 ───────────────────────────────────────────────
N_CLUSTERS = 20
print(f"\nClustering {D} feature vectors into {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(W_norm)
print(f"  ✓ KMeans done")

# ── UMAP 降维 ─────────────────────────────────────────────────
print("Running UMAP on decoder weight vectors...")
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=20,
    min_dist=0.05,
    metric='cosine',    # 用余弦距离，更适合 embedding direction vectors
    verbose=False
)
W_2d = reducer.fit_transform(W_norm)   # [4096, 2]
print(f"  ✓ UMAP done  shape={W_2d.shape}")

# ── 保存聚类结果 ──────────────────────────────────────────────
df_clusters = pd.DataFrame({
    'feature_id':   np.arange(D),
    'cluster':      cluster_labels,
    'act_rate':     act_rate,
    'hate_bias':    hate_bias,
    'umap_x':       W_2d[:, 0],
    'umap_y':       W_2d[:, 1],
})
# 每个 cluster 的平均 hate_bias（用于给 cluster 排序/着色）
cluster_hate = df_clusters.groupby('cluster')['hate_bias'].mean()
df_clusters['cluster_hate_mean'] = df_clusters['cluster'].map(cluster_hate)
df_clusters.to_csv(f"{OUT_DIR}/decoder_clusters.csv", index=False)
print(f"  ✓ decoder_clusters.csv")

# ── 已标注的 concept feature（来自 concept_analysis）────────────
ANNOTATED = {
    2837: 'Gender self-questioning',
    3874: 'Religion vs. homosexuality',
    2009: 'Conditional tolerance',
    3193: 'Trans identity anxiety',
    1796: 'Queer social dynamics',
    1381: 'Reddit anti-trans hate',
    3201: 'Corporate Pride mockery',
    2208: 'LGBTQ+ identity taxonomy',
    2448: 'Trans rights affirmation',
    2150: 'Trans civil rights analogy',
    1657: 'Anti-trans political rhetoric',
    114:  'Conservative opposition',
    962:  'Groomer framing',
    3224: 'Conservative anti-LGBTQ media',
    455:  'Anti-hate slogans',
    135:  'Trans activism',
    1210: 'Anti-trans child rhetoric',
    487:  "Don't Say Gay / FL law",
    3509: 'Conservative political memes',
    3836: 'LGBTQ rights vs. institution',
    3050: 'Say Gay discourse',
    1730: 'TERF debate',
    3905: 'Gender nonconformity mockery',
    2974: 'Media representation debate',
    2688: 'Personal coming-out',
    2053: 'Anti-trans science denial',
    719:  'Gender legislation memes',
    3193: 'Trans identity anxiety',
}

# ═══════════════════════════════════════════════════════════════
# 图5: 全局 UMAP，所有 4096 features，按 cluster 上色
# ═══════════════════════════════════════════════════════════════
print("\nPlotting fig5: full decoder UMAP...")

cmap = plt.cm.get_cmap('tab20', N_CLUSTERS)
fig, ax = plt.subplots(figsize=(11, 8))

# 所有点（小、半透明）
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    ax.scatter(
        W_2d[mask, 0], W_2d[mask, 1],
        s=4, alpha=.35, color=cmap(c), linewidths=0
    )

# 高激活 feature（rate > 0.03）用大点标出
high_mask = act_rate > 0.03
ax.scatter(
    W_2d[high_mask, 0], W_2d[high_mask, 1],
    s=30, alpha=.8,
    c=[cmap(c) for c in cluster_labels[high_mask]],
    edgecolors='white', linewidths=.5, zorder=5
)

# 标注 annotated features
for fid, name in ANNOTATED.items():
    if fid < D:
        x, y = W_2d[fid]
        short = name[:22] + ('…' if len(name)>22 else '')
        ax.annotate(
            f"#{fid}", (x, y),
            fontsize=6.5, ha='center', va='bottom',
            xytext=(0, 4), textcoords='offset points',
            color='#333', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=.6)
        )

# cluster 中心标签
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    cx, cy = W_2d[mask].mean(axis=0)
    ax.text(cx, cy, str(c), fontsize=8, ha='center', va='center',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=cmap(c), ec='none', alpha=.8))

ax.set_title('InterVLM: SAE Decoder Weight Vector Space\n'
             '(each point = 1 feature, colored by cluster, large dots = high activation rate)',
             fontsize=11)
ax.set_xlabel('UMAP-1 (cosine metric)'); ax.set_ylabel('UMAP-2')
ax.grid(True, alpha=.1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_decoder_umap.png", dpi=150)
plt.close()
print(f"  ✓ fig5_decoder_umap.png")

# ═══════════════════════════════════════════════════════════════
# 图6: 只展示标注 feature，按 hate_bias 上色，标注 concept 名
# ═══════════════════════════════════════════════════════════════
print("Plotting fig6: annotated features only...")

ann_ids   = [fid for fid in ANNOTATED if fid < D]
ann_x     = W_2d[ann_ids, 0]
ann_y     = W_2d[ann_ids, 1]
ann_bias  = hate_bias[ann_ids]
ann_rate  = act_rate[ann_ids]
ann_names = [ANNOTATED[fid] for fid in ann_ids]

fig, ax = plt.subplots(figsize=(13, 9))
sc = ax.scatter(
    ann_x, ann_y,
    c=ann_bias, cmap='RdBu_r', vmin=-.07, vmax=.07,
    s=80 + ann_rate / ann_rate.max() * 300,
    edgecolors='gray', linewidths=.5, alpha=.9, zorder=3
)
plt.colorbar(sc, ax=ax, label='Hate bias', shrink=.7)

for i, (fid, name) in enumerate(zip(ann_ids, ann_names)):
    ax.annotate(
        f"#{fid} {name}", (ann_x[i], ann_y[i]),
        fontsize=7.5,
        xytext=(5, 5), textcoords='offset points',
        color='#222',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#ccc', alpha=.75)
    )

ax.set_title('InterVLM: Annotated SAE Features in Decoder Space\n'
             '(color = hate bias, size = activation rate)',
             fontsize=11)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.grid(True, alpha=.1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_decoder_annotated.png", dpi=150)
plt.close()
print(f"  ✓ fig6_decoder_annotated.png")

# ═══════════════════════════════════════════════════════════════
# 图7: Cluster-level 摘要 — 每个 cluster 的 hate_bias 分布
# ═══════════════════════════════════════════════════════════════
print("Plotting fig7: cluster hate bias summary...")

cluster_stats = df_clusters.groupby('cluster').agg(
    n_features=('feature_id', 'count'),
    mean_hate_bias=('hate_bias', 'mean'),
    mean_act_rate=('act_rate', 'mean'),
    n_high=('act_rate', lambda x: (x > 0.02).sum()),
).reset_index().sort_values('mean_hate_bias')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: cluster hate_bias 排序
colors_bar = [
    '#E24B4A' if v > 0.01 else '#378ADD' if v < -0.01 else '#888780'
    for v in cluster_stats['mean_hate_bias']
]
bars = axes[0].barh(
    [f"Cluster {int(r.cluster):2d}" for _, r in cluster_stats.iterrows()],
    cluster_stats['mean_hate_bias'],
    color=colors_bar, alpha=.8, edgecolor='none'
)
axes[0].axvline(0, color='gray', lw=.8, ls='--')
axes[0].set_xlabel('Mean Hate Bias')
axes[0].set_title('Cluster-level Hate Bias')
axes[0].bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
axes[0].tick_params(axis='y', labelsize=8)

# 右图: cluster size vs mean_act_rate
axes[1].scatter(
    cluster_stats['n_features'],
    cluster_stats['mean_act_rate'],
    c=cluster_stats['mean_hate_bias'],
    cmap='RdBu_r', vmin=-.07, vmax=.07,
    s=80, edgecolors='gray', linewidths=.5, alpha=.9
)
for _, r in cluster_stats.iterrows():
    axes[1].annotate(f"C{int(r.cluster)}", (r.n_features, r.mean_act_rate),
                     fontsize=7, ha='center', va='bottom',
                     xytext=(0, 3), textcoords='offset points')
axes[1].set_xlabel('Number of features in cluster')
axes[1].set_ylabel('Mean activation rate')
axes[1].set_title('Cluster Size vs. Activity')
axes[1].grid(True, alpha=.15)

plt.suptitle('InterVLM: SAE Feature Cluster Summary', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_cluster_summary.png", dpi=150)
plt.close()
print(f"  ✓ fig7_cluster_summary.png")

# ── 打印各 cluster 的 top feature ────────────────────────────
print("\n=== Cluster 摘要（按 hate_bias 排序）===")
for _, row in cluster_stats.iterrows():
    c = int(row.cluster)
    top_feat = df_clusters[df_clusters.cluster==c].nlargest(3,'act_rate')['feature_id'].tolist()
    ann_in_cluster = [f"#{fid}({ANNOTATED[fid][:20]})"
                      for fid in top_feat if fid in ANNOTATED]
    print(f"  Cluster {c:2d}  n={int(row.n_features):3d}  "
          f"hate_bias={row.mean_hate_bias:+.3f}  "
          f"act_rate={row.mean_act_rate:.3f}  "
          f"annotated={ann_in_cluster}")

print(f"\n全部完成！图片保存在: {FIG_DIR}")
