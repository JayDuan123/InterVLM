"""
umap_compare.py
两张 UMAP 对比：
  Fig A: 原始 CLIP embedding h，按 top-20 dominant 维度上色
  Fig B: SAE latent z，按 top-20 dominant feature 上色
颜色方案：
  hate=1 相关 → 红色系（深浅按 hate_bias 排序）
  hate=0 相关 → 蓝色系
  stance       → 绿色系
  humour       → 橙色系
  target       → 紫色系
  其余         → 灰色
"""

import os, torch, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import normalize

try:
    import umap as umap_lib
except ImportError:
    raise ImportError("pip install umap-learn")

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
DATA    = f"{BASE}/MemeCLIP/data"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading data...")
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")
tr_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
te_lat = torch.load(f"{OUT_DIR}/test_latents.pt")

h_all   = torch.cat([tr_emb['embeddings'], te_emb['embeddings']]).float().numpy()
z_all   = torch.cat([tr_lat['z'],          te_lat['z']         ]).numpy()
hate    = torch.cat([tr_emb['hate'],        te_emb['hate']      ]).numpy()
stance  = torch.cat([tr_emb['stance'],      te_emb['stance']    ]).numpy()
humour  = torch.cat([tr_emb['humour'],      te_emb['humour']    ]).numpy()
target  = torch.cat([tr_emb['target'],      te_emb['target']    ]).numpy()

df_tr = pd.read_csv(f"{DATA}/train.csv")
df_te = pd.read_csv(f"{DATA}/test.csv")
df    = pd.concat([df_tr, df_te], ignore_index=True)

mean_h = h_all.mean(0); std_h = h_all.std(0).clip(1e-6)
h_norm = (h_all - mean_h) / std_h

# ── Fisher p-value 判断每个 feature/维度偏向哪个 concept ──────
pval_df = pd.read_csv(f"{OUT_DIR}/concept_mapping_pval.csv")
concept_cols = [c for c in pval_df.columns if c != 'feature_id']
pval_mat = pval_df[concept_cols].values   # [4096, 8]

def get_concept_category(pvals_row, threshold=0.05):
    """返回 (大类, 子类index)，用于颜色映射"""
    sig = pvals_row < threshold
    # 优先级：hate > stance > humour > target
    if sig[0]: return ('hate',    0)   # hate=1
    if sig[1]: return ('benign',  1)   # hate=0
    if sig[4]: return ('stance',  4)   # stance=support
    if sig[5]: return ('stance',  5)   # stance=oppose
    if sig[6]: return ('humour',  6)   # humour=1
    if sig[7]: return ('humour',  7)   # humour=0
    if sig[2]: return ('target',  2)   # target=community
    if sig[3]: return ('target',  3)   # target=individual
    return ('other', -1)

# 大类颜色（基础色）
BASE_COLORS = {
    'hate':   '#D32F2F',   # 红
    'benign': '#1565C0',   # 蓝
    'stance': '#2E7D32',   # 绿
    'humour': '#E65100',   # 橙
    'target': '#6A1B9A',   # 紫
    'other':  '#CCCCCC',   # 灰
}

# ── UMAP 工具 ─────────────────────────────────────────────────
def get_umap(data, cache, metric='euclidean', n_neighbors=30, min_dist=0.2):
    if os.path.exists(cache):
        print(f"  Loading cached UMAP: {os.path.basename(cache)}")
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

# ── 获取 top-20 dominant 并分配颜色 ──────────────────────────
def get_top20_colors(activations, pval_matrix, top_n=20):
    """
    activations: [N, D]
    返回每个样本的颜色（对应其 dominant feature 的概念颜色）
    """
    dom_feat   = activations.argmax(axis=1)           # [N] dominant feature index
    dom_counts = pd.Series(dom_feat).value_counts()
    top20_ids  = dom_counts.head(top_n).index.tolist()

    # 给每个 top-20 feature 分配颜色
    feat_colors = {}
    cat_count   = {'hate':0,'benign':0,'stance':0,'humour':0,'target':0,'other':0}

    for fid in top20_ids:
        cat, _ = get_concept_category(pval_matrix[fid])
        base   = BASE_COLORS[cat]
        # 深浅：同类里第几个，越早越深
        n = cat_count[cat]
        # 用 alpha/lightness 区分深浅（0=最深, 4=最浅）
        factor = 1.0 - n * 0.15
        factor = max(factor, 0.4)
        rgb    = mcolors.to_rgb(base)
        # 调亮：向白色插值
        light  = tuple(c + (1-c)*(1-factor) for c in rgb)
        feat_colors[fid] = light
        cat_count[cat]  += 1

    return top20_ids, feat_colors, dom_feat

# ── 画图函数 ──────────────────────────────────────────────────
def plot_concept_umap(coords_2d, activations, pval_matrix,
                      title, out_path, top_n=20):
    mask      = remove_outliers(coords_2d)
    coords    = coords_2d[mask]
    acts      = activations[mask]

    top20_ids, feat_colors, dom_feat = get_top20_colors(activations, pval_matrix, top_n)
    dom_clean = dom_feat[mask]

    fig, ax = plt.subplots(figsize=(12, 9))

    # 灰色背景
    bg = np.array([f not in feat_colors for f in dom_clean])
    ax.scatter(coords[bg,0], coords[bg,1],
               c='#E0E0E0', s=3, alpha=.15, linewidths=0, zorder=1)

    # 彩色前景，按大类分组画，方便 legend
    plotted_cats = {}
    for i in range(len(coords)):
        fid = dom_clean[i]
        if fid not in feat_colors: continue
        col = feat_colors[fid]
        cat, _ = get_concept_category(pval_matrix[fid])
        label  = None
        if cat not in plotted_cats:
            label = cat
            plotted_cats[cat] = True
        ax.scatter(coords[i,0], coords[i,1],
                   c=[col], s=22, alpha=.88, linewidths=0,
                   zorder=3, label=label)

    # 标注每个 cluster 中心
    for fid in top20_ids:
        pts = coords[dom_clean == fid]
        if len(pts) < 3: continue
        cx, cy = pts.mean(axis=0)
        cat, _ = get_concept_category(pval_matrix[fid])
        ax.annotate(
            f"#{fid}",
            (cx, cy), fontsize=7, ha='center', va='center',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2',
                      fc=BASE_COLORS[cat], ec='none', alpha=.75),
            zorder=5
        )

    # Legend（大类）
    legend_handles = [
        plt.scatter([], [], c=BASE_COLORS[cat], s=40, label=cat.capitalize())
        for cat in ['hate','benign','stance','humour','target']
        if cat in plotted_cats
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=9, framealpha=.88, title='Concept category', title_fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.grid(True, alpha=.07)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")

# ── 生成 UMAP 坐标 ────────────────────────────────────────────
print("\nGenerating UMAPs...")

h_2d = get_umap(h_norm, f"{OUT_DIR}/umap_h_coords.npy",
                metric='euclidean', n_neighbors=15, min_dist=0.1)

bias_all = (z_all[hate==1]>0).mean(0) - (z_all[hate==0]>0).mean(0)
top200   = np.argsort(np.abs(bias_all))[::-1][:200]
z_sub    = normalize(z_all[:,top200], norm='l2')
z_2d     = get_umap(z_sub, f"{OUT_DIR}/umap_z_coords.npy",
                    metric='cosine', n_neighbors=30, min_dist=0.2)

# ── 图A: h 空间（用 h 本身的维度做 dominant）────────────────
print("\nPlotting Fig A: h space...")

# h 的 top-20 dominant 维度需要单独的 pval_matrix（用 h 维度而非 SAE feature）
# 这里用 SAE feature 的 pval 作为颜色参考，但 dominant 是 h 的维度
# 更直接的做法：用 h_norm 的绝对值最大的维度
h_abs  = np.abs(h_norm)   # [N, 768]

# 为 h 的每个维度计算 hate_bias（用作颜色）
h_hate_rate = h_abs[hate==1].mean(0)
h_ben_rate  = h_abs[hate==0].mean(0)
h_bias      = h_hate_rate - h_ben_rate   # [768]

# 构造 h 维度的 pval_matrix（简化版：只用 hate 维度的 bias 分类）
# 768 维，每个维度判断偏向哪个 concept
h_pval_proxy = np.ones((768, 8))   # 默认不显著

# 用 hate_bias 判断大类（bias>0.01 → hate, <-0.01 → benign）
for dim in range(768):
    b = h_bias[dim]
    # 按 stance
    st_rate = h_abs[stance==2].mean(0)[dim] - h_abs[stance==1].mean(0)[dim]
    hu_rate = h_abs[humour==1].mean(0)[dim] - h_abs[humour==0].mean(0)[dim]
    if b > 0.005:
        h_pval_proxy[dim, 0] = 0.01   # hate=1
    elif b < -0.005:
        h_pval_proxy[dim, 1] = 0.01   # hate=0
    elif abs(st_rate) > 0.005:
        h_pval_proxy[dim, 4 if st_rate<0 else 5] = 0.01
    elif abs(hu_rate) > 0.005:
        h_pval_proxy[dim, 6 if hu_rate>0 else 7] = 0.01

# ── 图A: h 空间（按真实标签上色）────────────────────────────
print("\nPlotting Fig A: h space colored by true labels...")

mask_h    = remove_outliers(h_2d)
coords_h  = h_2d[mask_h]
hate_c    = hate[mask_h]
stance_c  = stance[mask_h]
humour_c  = humour[mask_h]

# 每个样本分配一个大类颜色（优先级：hate > stance > humour > other）
# hate=1 → 红, hate=0+stance=oppose → 深绿, hate=0+stance=support → 蓝,
# humour=1 → 橙, 其余 → 灰
def assign_label_color(h, st, hu):
    if h == 1:   return '#D32F2F'   # hate → 红
    if st == 2:  return '#1B5E20'   # stance=oppose → 深绿
    if st == 1:  return '#1565C0'   # stance=support → 蓝
    if hu == 1:  return '#E65100'   # humour=1 → 橙
    return '#9E9E9E'                # 其余 → 灰

colors_h = [assign_label_color(h, st, hu)
            for h, st, hu in zip(hate_c, stance_c, humour_c)]

fig, ax = plt.subplots(figsize=(12, 9))
for color, label in [
    ('#D32F2F', 'Hate'),
    ('#1B5E20', 'Stance=Oppose'),
    ('#1565C0', 'Stance=Support'),
    ('#E65100', 'Humour'),
    ('#9E9E9E', 'Other'),
]:
    mask = np.array([c == color for c in colors_h])
    if mask.sum() == 0: continue
    ax.scatter(coords_h[mask, 0], coords_h[mask, 1],
               c=color, s=8 if color != '#9E9E9E' else 4,
               alpha=.6 if color != '#9E9E9E' else .2,
               linewidths=0, label=label, zorder=3 if color != '#9E9E9E' else 1)

ax.legend(loc='upper left', fontsize=9, framealpha=.88,
          markerscale=2, title='Label category', title_fontsize=8)
ax.set_title('UMAP: CLIP Fused Representation (h)\n'
             'colored by true label (hate / stance / humour)', fontsize=12)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.grid(True, alpha=.07)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/figA_umap_h_concept.png", dpi=150)
plt.close()
print("  ✓ figA_umap_h_concept.png")

# ── 图B: z 空间 ───────────────────────────────────────────────
print("Plotting Fig B: SAE z space...")
plot_concept_umap(
    z_2d,
    z_all[:, top200],   # 只用 top200 bias features
    pval_mat[top200],   # 对应的 pval
    title='UMAP: SAE Latent Space (z, top-200 bias features)\nTop-20 dominant features colored by concept category',
    out_path=f"{FIG_DIR}/figB_umap_z_concept.png",
    top_n=20
)

print("\n完成！")
print(f"  figA_umap_h_concept.png")
print(f"  figB_umap_z_concept.png")
