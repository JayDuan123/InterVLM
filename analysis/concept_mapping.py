"""
concept_mapping.py  —  InterVLM Concept Mapping
对 SAE feature × 13 个 concept 做 Fisher's exact test
标签直接从 embedding 文件读取（支持合并数据集）
"""

import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载 latent codes ─────────────────────────────────────────
print("Loading data...")
train_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
test_lat  = torch.load(f"{OUT_DIR}/test_latents.pt")
z_all = np.concatenate([train_lat['z'].numpy(), test_lat['z'].numpy()])

# ── 从 embedding 文件读取所有标签 ─────────────────────────────
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")

def get_label(key, na_val=-1):
    n_tr = len(tr_emb['embeddings'])
    n_te = len(te_emb['embeddings'])
    if key in tr_emb:
        tr_l = tr_emb[key].numpy()
    else:
        tr_l = np.full(n_tr, na_val)
    if key in te_emb:
        te_l = te_emb[key].numpy()
    else:
        te_l = np.full(n_te, na_val)
    return np.concatenate([tr_l, te_l])

hate        = get_label('hate')
target      = get_label('target')
stance      = get_label('stance')
humor       = get_label('humor')
offensive   = get_label('offensive')
sentiment   = get_label('sentiment', na_val=-99)
motivational= get_label('motivational')
sarcasm     = get_label('sarcasm')

N = len(hate)
print(f"  总样本数: {N}  feature 数: {z_all.shape[1]}")

# 确认维度一致
assert N == z_all.shape[0], f"样本数不一致: labels={N}, z={z_all.shape[0]}"

# ── 定义 13 个 concept ────────────────────────────────────────
CONCEPTS = {
    'hate=1':           hate == 1,
    'hate=0':           hate == 0,
    'offensive>=2':     offensive >= 2,
    'offensive=0':      offensive == 0,
    'stance=support':   stance == 1,
    'stance=oppose':    stance == 2,
    'humor>=2':         humor >= 2,
    'humor=0':          humor == 0,
    'sentiment>=1':     sentiment >= 1,
    'sentiment<=-1':    sentiment <= -1,
    'sarcasm>=1':       sarcasm >= 1,
    'motivational=1':   motivational == 1,
    'target=community': target == 2,
}

# 打印每个 concept 的样本数
print("\n  Concept 样本分布:")
for k, v in CONCEPTS.items():
    print(f"    {k:22s}: {v.sum():6d} 条")

concept_names = list(CONCEPTS.keys())
N_CONCEPTS    = len(concept_names)
N_FEATURES    = z_all.shape[1]

# ── Fisher's exact test ───────────────────────────────────────
print(f"\nRunning Fisher's exact test ({N_FEATURES} features × {N_CONCEPTS} concepts)...")
print("  (this may take ~5 minutes)")

pval_matrix   = np.ones((N_FEATURES, N_CONCEPTS))
odds_matrix   = np.ones((N_FEATURES, N_CONCEPTS))
active_binary = (z_all > 0)

for j, cname in enumerate(concept_names):
    concept_mask = CONCEPTS[cname].astype(bool)
    for i in range(N_FEATURES):
        feat_active = active_binary[:, i]
        a = ( feat_active &  concept_mask).sum()
        b = ( feat_active & ~concept_mask).sum()
        c = (~feat_active &  concept_mask).sum()
        d = (~feat_active & ~concept_mask).sum()
        if a + b == 0 or a + c == 0:
            continue
        odds, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        pval_matrix[i, j] = p
        odds_matrix[i, j] = odds

    _, pval_corr, _, _ = multipletests(pval_matrix[:, j], method='fdr_bh')
    pval_matrix[:, j]  = pval_corr
    sig = (pval_corr < 0.05).sum()
    print(f"  {cname:25s}  significant features: {sig}")

# ── 保存 ──────────────────────────────────────────────────────
pd.DataFrame(pval_matrix, columns=concept_names).assign(
    feature_id=np.arange(N_FEATURES)
).to_csv(f"{OUT_DIR}/concept_mapping_pval.csv", index=False)

pd.DataFrame(odds_matrix, columns=concept_names).assign(
    feature_id=np.arange(N_FEATURES)
).to_csv(f"{OUT_DIR}/concept_mapping_odds.csv", index=False)
print(f"\n✓ concept_mapping_pval.csv / odds.csv 保存完成")

# ── 摘要 ──────────────────────────────────────────────────────
print("\n=== Concept Mapping 摘要 ===")
total_sig = 0
for j, cname in enumerate(concept_names):
    sig = (pval_matrix[:, j] < 0.05).sum()
    total_sig += sig
    print(f"  {cname:25s}: {sig:4d} significant (FDR<0.05)")
print(f"\n  总计: {total_sig} (feature, concept) 显著对应")

# ── Fig 8: Heatmap ────────────────────────────────────────────
print("\nPlotting heatmap...")
sig_mask  = (pval_matrix < 0.05).any(axis=1)
sig_ids   = np.where(sig_mask)[0]
print(f"  至少对一个 concept 显著的 feature 数: {len(sig_ids)}")

logp      = -np.log10(pval_matrix[sig_ids] + 1e-10).clip(0, 10)
order     = np.argsort(logp[:, 0])[::-1][:50]
logp_plot = logp[order]
fids_plot = sig_ids[order]

fig, ax = plt.subplots(figsize=(14, 14))
im = ax.imshow(logp_plot, aspect='auto', cmap='YlOrRd', vmin=0, vmax=6)
plt.colorbar(im, ax=ax, label='-log10(FDR p)', shrink=.6)
ax.set_xticks(range(N_CONCEPTS))
ax.set_xticklabels(concept_names, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(fids_plot)))
ax.set_yticklabels([f"#{fid}" for fid in fids_plot], fontsize=7.5)
ax.set_title('InterVLM: SAE Feature × Concept Mapping (13 concepts, FDR<0.05)', fontsize=12)
for i in range(len(fids_plot)):
    for j in range(N_CONCEPTS):
        if logp_plot[i, j] > 1.3:
            ax.text(j, i, '★', ha='center', va='center',
                    fontsize=6.5, color='white' if logp_plot[i,j]>3 else '#333')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_concept_heatmap.png", dpi=150)
plt.close()
print("✓ fig8_concept_heatmap.png")

# ── Fig 9: Top-10 per concept ─────────────────────────────────
print("Plotting top features per concept...")
n_cols = 4
n_rows = (N_CONCEPTS + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

COLORS = {
    'hate':        '#E24B4A',
    'offensive':   '#EF9F27',
    'stance':      '#2E7D32',
    'humor':       '#E65100',
    'sentiment':   '#1565C0',
    'sarcasm':     '#6A1B9A',
    'motivational':'#00838F',
    'target':      '#AD1457',
}

for j, cname in enumerate(concept_names):
    ax = axes[j]
    top10_idx  = np.argsort(pval_matrix[:, j])[:10]
    top10_logp = -np.log10(pval_matrix[top10_idx, j] + 1e-10).clip(0, 10)
    cat  = cname.split('=')[0].split('>')[0].split('<')[0]
    color = COLORS.get(cat, '#888780')
    bars = ax.barh([f"#{i}" for i in top10_idx], top10_logp,
                   color=color, alpha=.75, edgecolor='none')
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=7.5)
    ax.axvline(1.3, color='gray', lw=.8, ls='--', alpha=.5)
    ax.set_xlim(0, max(top10_logp.max() * 1.2, 2))
    ax.set_title(cname, fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('-log10(FDR p)', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=.15)

for j in range(N_CONCEPTS, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('InterVLM: Top-10 SAE Features per Concept', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig9_top_features_per_concept.png", dpi=150)
plt.close()
print("✓ fig9_top_features_per_concept.png")
print("\n完成！")
