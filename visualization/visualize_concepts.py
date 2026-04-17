"""
visualize_concepts.py  —  InterVLM Concept Visualization
生成四张图:
  1. Feature scatter plot (hate_bias vs activation_rate)
  2. Top hate-biased / benign-biased features bar chart
  3. UMAP of h (if already generated)
  4. UMAP of z (if already generated)
所有图保存到 sae_outputs/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT_DIR  = "/projectnb/cepinet/users/Jay/InterVLP/sae_outputs"
FIG_DIR  = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 手动标注的 concept 数据（来自 top_features.txt 分析）────────
CONCEPTS = [
    dict(id=2837, rate=.078, bias=-.067, type='benign',    concept='Gender self-questioning'),
    dict(id=3874, rate=.060, bias=+.022, type='mild-hate', concept='Religion vs. homosexuality'),
    dict(id=2009, rate=.058, bias=-.029, type='benign',    concept='Conditional tolerance'),
    dict(id=3193, rate=.053, bias=-.044, type='benign',    concept='Trans identity anxiety'),
    dict(id=1796, rate=.052, bias=-.012, type='neutral',   concept='Queer social dynamics'),
    dict(id=1381, rate=.049, bias=+.048, type='hate',      concept='Reddit anti-trans hate'),
    dict(id=3201, rate=.048, bias=+.034, type='hate',      concept='Corporate Pride mockery'),
    dict(id=3905, rate=.047, bias=+.006, type='neutral',   concept='Gender nonconformity mockery'),
    dict(id=2208, rate=.046, bias=-.047, type='benign',    concept='LGBTQ+ identity taxonomy'),
    dict(id=2053, rate=.046, bias=+.016, type='mild-hate', concept='Anti-trans science denial'),
    dict(id=2974, rate=.044, bias=-.006, type='neutral',   concept='Media representation debate'),
    dict(id=3050, rate=.043, bias=-.021, type='neutral',   concept='Say Gay discourse'),
    dict(id=2688, rate=.043, bias=-.011, type='neutral',   concept='Personal coming-out'),
    dict(id=455,  rate=.043, bias=-.013, type='benign',    concept='Anti-hate slogans'),
    dict(id=2448, rate=.043, bias=-.040, type='benign',    concept='Trans rights affirmation'),
    dict(id=2150, rate=.042, bias=-.043, type='benign',    concept='Trans civil rights analogy'),
    dict(id=1730, rate=.042, bias=+.006, type='neutral',   concept='TERF debate'),
    dict(id=135,  rate=.042, bias=-.027, type='benign',    concept='Trans activism / protest'),
    dict(id=1657, rate=.042, bias=+.055, type='hate',      concept='Anti-trans political rhetoric'),
    dict(id=114,  rate=.038, bias=+.045, type='hate',      concept='Conservative opposition'),
    dict(id=962,  rate=.039, bias=+.038, type='hate',      concept='Groomer framing'),
    dict(id=3224, rate=.039, bias=+.038, type='hate',      concept='Conservative anti-LGBTQ media'),
    dict(id=1210, rate=.035, bias=+.030, type='hate',      concept='Anti-trans child rhetoric'),
    dict(id=719,  rate=.027, bias=+.032, type='hate',      concept='Gender legislation memes'),
    dict(id=487,  rate=.029, bias=+.030, type='hate',      concept="Don't Say Gay / FL law"),
    dict(id=3509, rate=.025, bias=+.031, type='hate',      concept='Conservative political memes'),
    dict(id=3836, rate=.041, bias=-.022, type='neutral',   concept='LGBTQ rights vs. institution'),
]
df = pd.DataFrame(CONCEPTS)

TYPE_COLOR = {
    'hate':      '#E24B4A',
    'mild-hate': '#EF9F27',
    'neutral':   '#888780',
    'benign':    '#378ADD',
}
TYPE_LABEL = {
    'hate':      'Hate-biased',
    'mild-hate': 'Mild hate',
    'neutral':   'Neutral',
    'benign':    'Benign-biased',
}

# ════════════════════════════════════════════════════════════════
# 图1: Feature scatter — hate_bias vs activation_rate
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

for _, row in df.iterrows():
    color = TYPE_COLOR[row['type']]
    size  = 80 + (row['rate'] - .024) / (.078 - .024) * 400
    ax.scatter(row['bias'], row['rate'], s=size,
               color=color, alpha=.75, edgecolors=color, linewidths=.8, zorder=3)

# 标注所有 feature id
for _, row in df.iterrows():
    ax.annotate(f"#{row['id']}", (row['bias'], row['rate']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points', color='#444')

ax.axvline(0, color='gray', lw=.8, ls='--', alpha=.5)
ax.set_xlabel('Hate Bias  (hate activation rate − benign activation rate)', fontsize=11)
ax.set_ylabel('Activation Rate', fontsize=11)
ax.set_title('InterVLM: SAE Feature Map\n(size = activation rate)', fontsize=13)

legend_handles = [
    mpatches.Patch(color=TYPE_COLOR[t], label=TYPE_LABEL[t])
    for t in ['hate', 'mild-hate', 'neutral', 'benign']
]
ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=.8)
ax.grid(True, alpha=.15)
ax.set_xlim(-.085, .075)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_feature_scatter.png", dpi=150)
plt.close()
print(f"✓ fig1_feature_scatter.png")

# ════════════════════════════════════════════════════════════════
# 图2: Top hate / benign features — horizontal bar chart
# ════════════════════════════════════════════════════════════════
top_hate   = df.nlargest(8, 'bias').sort_values('bias')
top_benign = df.nsmallest(8, 'bias').sort_values('bias', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, subset, title, color in [
    (axes[0], top_hate,   'Top Hate-Biased Features',   '#E24B4A'),
    (axes[1], top_benign, 'Top Benign-Biased Features', '#378ADD'),
]:
    labels = [f"#{int(r.id)}\n{r.concept}" for _, r in subset.iterrows()]
    values = subset['bias'].abs().values
    bars = ax.barh(labels, values, color=color, alpha=.75, edgecolor=color)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_xlabel('|Hate Bias|', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=.2)
    ax.set_xlim(0, values.max() * 1.25)

plt.suptitle('InterVLM: Feature Polarization', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_feature_polarity.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ fig2_feature_polarity.png")

# ════════════════════════════════════════════════════════════════
# 图3 & 4: UMAP (if .pt files already exist from concept_analysis.py)
# ════════════════════════════════════════════════════════════════
import torch

try:
    import umap as umap_lib

    train_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
    test_lat  = torch.load(f"{OUT_DIR}/test_latents.pt")
    norm_stats= torch.load(f"{OUT_DIR}/norm_stats.pt")

    h_all   = torch.cat([train_lat['embeddings'], test_lat['embeddings']], dim=0).float().numpy()
    z_all   = torch.cat([train_lat['z'],          test_lat['z']         ], dim=0).numpy()
    lbl_all = torch.cat([train_lat['labels'],      test_lat['labels']    ]).numpy()

    mean = norm_stats['mean'].numpy()
    std  = norm_stats['std'].numpy()
    h_norm = (h_all - mean) / (std + 1e-6)

    reducer = umap_lib.UMAP(n_components=2, random_state=42,
                             n_neighbors=15, min_dist=0.1, verbose=False)

    # 图3: UMAP of h
    print("Running UMAP on h...")
    h_2d = reducer.fit_transform(h_norm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(h_2d[lbl_all==0, 0], h_2d[lbl_all==0, 1],
                    c='#378ADD', s=6, alpha=.5, label='Benign')
    ax.scatter(h_2d[lbl_all==1, 0], h_2d[lbl_all==1, 1],
               c='#E24B4A', s=6, alpha=.5, label='Hate')
    ax.set_title('UMAP: MemeCLIP Fused Representation (h)', fontsize=12)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=.1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig3_umap_h.png", dpi=150)
    plt.close()
    print(f"✓ fig3_umap_h.png")

    # 图4: UMAP of z (top-200 active features)
    print("Running UMAP on z...")
    act_rate = (z_all > 0).mean(axis=0)
    top200   = np.argsort(act_rate)[::-1][:200]
    z_sub    = z_all[:, top200]
    z_2d     = reducer.fit_transform(z_sub)
    fig, ax  = plt.subplots(figsize=(8, 6))
    ax.scatter(z_2d[lbl_all==0, 0], z_2d[lbl_all==0, 1],
               c='#378ADD', s=6, alpha=.5, label='Benign')
    ax.scatter(z_2d[lbl_all==1, 0], z_2d[lbl_all==1, 1],
               c='#E24B4A', s=6, alpha=.5, label='Hate')
    ax.set_title('UMAP: SAE Latent Space (z, top-200 features)', fontsize=12)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=.1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig4_umap_z.png", dpi=150)
    plt.close()
    print(f"✓ fig4_umap_z.png")

except ImportError:
    print("⚠ umap-learn 未安装，跳过 UMAP 图。运行: pip install umap-learn")
except Exception as e:
    print(f"⚠ UMAP 生成失败: {e}")

print(f"\n全部完成！图片保存在: {FIG_DIR}")
