"""
decoder_clustering_v2.py
fig6: annotated features UMAP，hate cluster 用虚线椭圆圈出
fig7: annotated features 按 hate_bias 分三组 bar chart
"""

import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError:
    raise ImportError("pip install umap-learn")

OUT_DIR = "/projectnb/cepinet/users/Jay/InterVLP/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载 SAE decoder weights ──────────────────────────────────
print("Loading SAE...")
ckpt     = torch.load(f"{OUT_DIR}/sae_model.pt", map_location='cpu')
decoder_w = ckpt['state_dict']['decoder.weight'].numpy()   # [1024, 4096]
W        = decoder_w.T                                     # [4096, 1024]
W_norm   = normalize(W, norm='l2')
D        = W.shape[0]

# ── 激活统计 ──────────────────────────────────────────────────
train_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
test_lat  = torch.load(f"{OUT_DIR}/test_latents.pt")
z_all   = np.concatenate([train_lat['z'].numpy(), test_lat['z'].numpy()], axis=0)
lbl_all = np.concatenate([train_lat['labels'].numpy(), test_lat['labels'].numpy()])
act_rate  = (z_all > 0).mean(axis=0)
hate_rate = (z_all[lbl_all==1] > 0).mean(axis=0)
ben_rate  = (z_all[lbl_all==0] > 0).mean(axis=0)
hate_bias = hate_rate - ben_rate

# ── UMAP（复用已有结果或重新算）─────────────────────────────
umap_cache = f"{OUT_DIR}/decoder_umap_coords.npy"
if os.path.exists(umap_cache):
    print("Loading cached UMAP coords...")
    W_2d = np.load(umap_cache)
else:
    print("Running UMAP (this takes ~2 min)...")
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=20, min_dist=0.05,
                        metric='cosine', verbose=False)
    W_2d = reducer.fit_transform(W_norm)
    np.save(umap_cache, W_2d)
    print("  ✓ cached")

# ── Annotated features ────────────────────────────────────────
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
}

ann_ids   = [fid for fid in ANNOTATED if fid < D]
ann_x     = W_2d[ann_ids, 0]
ann_y     = W_2d[ann_ids, 1]
ann_bias  = hate_bias[ann_ids]
ann_rate  = act_rate[ann_ids]
ann_names = [ANNOTATED[fid] for fid in ann_ids]

# ── 分三组 ────────────────────────────────────────────────────
def group(bias):
    if bias > 0.02:  return 'hate'
    if bias < -0.02: return 'benign'
    return 'neutral'

groups = [group(b) for b in ann_bias]
GROUP_COLOR = {'hate': '#E24B4A', 'neutral': '#888780', 'benign': '#378ADD'}

# ═══════════════════════════════════════════════════════════════
# FIG 6: UMAP + 虚线椭圆圈出 hate / benign cluster
# ═══════════════════════════════════════════════════════════════
def fit_ellipse(xs, ys, pad=0.25):
    """返回椭圆中心 cx,cy 和半轴 rx,ry（加 padding）"""
    cx, cy = xs.mean(), ys.mean()
    rx = (xs.max() - xs.min()) / 2 + pad
    ry = (ys.max() - ys.min()) / 2 + pad
    return cx, cy, rx, ry

fig, ax = plt.subplots(figsize=(13, 9))

# 底层：全部 feature（灰色小点）
ax.scatter(W_2d[:, 0], W_2d[:, 1],
           s=2, alpha=.12, color='#aaaaaa', linewidths=0, zorder=1)

# 标注 feature 点
for i, (fid, name) in enumerate(zip(ann_ids, ann_names)):
    g = groups[i]
    c = GROUP_COLOR[g]
    size = 60 + ann_rate[i] / ann_rate.max() * 280
    ax.scatter(ann_x[i], ann_y[i], s=size, color=c,
               edgecolors='white', linewidths=.8, alpha=.92, zorder=4)

# 标注文字
for i, (fid, name) in enumerate(zip(ann_ids, ann_names)):
    ax.annotate(
        f"#{fid} {name}",
        (ann_x[i], ann_y[i]),
        fontsize=7.2,
        xytext=(5, 4), textcoords='offset points',
        color='#222',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#cccccc', alpha=.78),
        zorder=5
    )

# 虚线椭圆：hate cluster（bias > 0.02）
hate_mask = np.array([g == 'hate' for g in groups])
if hate_mask.sum() >= 2:
    cx, cy, rx, ry = fit_ellipse(ann_x[hate_mask], ann_y[hate_mask], pad=0.30)
    ellipse = Ellipse((cx, cy), width=rx*2, height=ry*2,
                      edgecolor='#E24B4A', facecolor='none',
                      linestyle='--', linewidth=1.6, alpha=.8, zorder=3)
    ax.add_patch(ellipse)
    ax.text(cx + rx - 0.05, cy + ry + 0.08, 'Hate cluster',
            color='#E24B4A', fontsize=8.5, fontweight='bold', ha='right')

# 虚线椭圆：benign cluster（bias < -0.02）
ben_mask = np.array([g == 'benign' for g in groups])
if ben_mask.sum() >= 2:
    cx, cy, rx, ry = fit_ellipse(ann_x[ben_mask], ann_y[ben_mask], pad=0.30)
    ellipse = Ellipse((cx, cy), width=rx*2, height=ry*2,
                      edgecolor='#378ADD', facecolor='none',
                      linestyle='--', linewidth=1.6, alpha=.8, zorder=3)
    ax.add_patch(ellipse)
    ax.text(cx - rx + 0.05, cy + ry + 0.08, 'Benign cluster',
            color='#378ADD', fontsize=8.5, fontweight='bold', ha='left')

# 图例
legend_handles = [
    mpatches.Patch(color='#E24B4A', label='Hate-biased  (bias > +0.02)'),
    mpatches.Patch(color='#888780', label='Neutral  (|bias| ≤ 0.02)'),
    mpatches.Patch(color='#378ADD', label='Benign-biased  (bias < −0.02)'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8.5,
          framealpha=.85, edgecolor='#ddd')

ax.set_title('InterVLM: Annotated SAE Features in Decoder Space\n'
             '(color = concept type, size = activation rate, dashed = clusters)',
             fontsize=11)
ax.set_xlabel('UMAP-1 (cosine metric)'); ax.set_ylabel('UMAP-2')
ax.grid(True, alpha=.1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_decoder_annotated_v2.png", dpi=150)
plt.close()
print("✓ fig6_decoder_annotated_v2.png")

# ═══════════════════════════════════════════════════════════════
# FIG 7: 三组 horizontal bar chart，按 hate_bias 排序
# ═══════════════════════════════════════════════════════════════
df_ann = pd.DataFrame({
    'feature_id': ann_ids,
    'concept':    ann_names,
    'hate_bias':  ann_bias,
    'act_rate':   ann_rate,
    'group':      groups,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 7),
                         gridspec_kw={'width_ratios': [1, 1, 1]})

group_order = ['hate', 'neutral', 'benign']
group_titles = {
    'hate':    'Hate-biased features\n(bias > +0.02)',
    'neutral': 'Neutral features\n(|bias| ≤ 0.02)',
    'benign':  'Benign-biased features\n(bias < −0.02)',
}

for ax, g in zip(axes, group_order):
    sub = df_ann[df_ann.group == g].sort_values('hate_bias',
                  ascending=(g != 'hate'))
    labels = [f"#{int(r.feature_id)}  {r.concept}" for _, r in sub.iterrows()]
    values = sub['hate_bias'].values
    sizes  = sub['act_rate'].values

    color = GROUP_COLOR[g]
    bars  = ax.barh(labels, values,
                    color=color, alpha=.75,
                    height=0.6, edgecolor='none')

    # 在 bar 末端标数值
    ax.bar_label(bars, fmt='%+.3f', padding=3, fontsize=7.5,
                 color='#333')

    # 用点大小叠加激活率信息
    for j, (label, v, s) in enumerate(zip(labels, values, sizes)):
        dot_x = v + (0.002 if v >= 0 else -0.002)
        ax.scatter(dot_x, j, s=40 + s/s.max()*120,
                   color=color, edgecolors='white',
                   linewidths=.6, alpha=.9, zorder=5)

    ax.axvline(0, color='gray', lw=.7, ls='--', alpha=.6)
    ax.set_title(group_titles[g], fontsize=10, color=color, fontweight='bold')
    ax.set_xlabel('Hate Bias', fontsize=9)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.grid(axis='x', alpha=.15)
    ax.set_xlim(
        min(values.min() - .01, -.005),
        max(values.max() + .015, .005)
    )

plt.suptitle('InterVLM: SAE Concept Features Grouped by Polarity\n'
             '(dot size = activation rate)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_concept_polarity_v2.png", dpi=150,
            bbox_inches='tight')
plt.close()
print("✓ fig7_concept_polarity_v2.png")
print(f"\n完成！保存在: {FIG_DIR}")
