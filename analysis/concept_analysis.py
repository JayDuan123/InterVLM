"""
concept_analysis.py  —  InterVLM Concept Analysis
1. Top-activating sample retrieval（每个 feature 找最强激活的 meme）
2. Feature 统计（激活频率、hate/benign 偏向）
3. UMAP 可视化（h 空间 + z 空间）
输出: sae_outputs/concept_report.csv   每个 feature 的统计摘要
      sae_outputs/umap_h.png           原始 embedding UMAP
      sae_outputs/umap_z.png           SAE latent UMAP
      sae_outputs/top_features.txt     top-20 feature 的 top-5 样本文本
"""

import os, sys, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, "/projectnb/cepinet/users/Jay/InterVLP/MemeCLIP/code")

BASE     = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR  = f"{BASE}/sae_outputs"
DATA_DIR = f"{BASE}/MemeCLIP/data"

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading latents and data...")
train_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
test_lat  = torch.load(f"{OUT_DIR}/test_latents.pt")

z_train   = train_lat['z'].numpy()            # [N_train, 4096]
z_test    = test_lat['z'].numpy()             # [N_test,  4096]
lbl_train = train_lat['labels'].numpy()       # [N_train]
lbl_test  = test_lat['labels'].numpy()
h_train   = train_lat['embeddings'].numpy()   # [N_train, 1024]
h_test    = test_lat['embeddings'].numpy()

# 合并 train+test 用于分析
z_all   = np.concatenate([z_train, z_test], axis=0)    # [N, 4096]
lbl_all = np.concatenate([lbl_train, lbl_test], axis=0)
h_all   = np.concatenate([h_train, h_test], axis=0)

# 原始 CSV（取 text 列用于展示）
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
df_test  = pd.read_csv(f"{DATA_DIR}/test.csv")
df_all   = pd.concat([df_train, df_test], ignore_index=True)

N, D = z_all.shape
print(f"  总样本: {N}  latent dim: {D}")
print(f"  hate={lbl_all.sum()}  benign={N - lbl_all.sum()}")

# ── 1. Feature 统计 ───────────────────────────────────────────
print("\nComputing feature statistics...")

activation_rate = (z_all > 0).mean(axis=0)         # [D] 每个 feature 激活频率
mean_activation = z_all.mean(axis=0)               # [D] 平均激活值

# 对每个 feature 算 hate bias = hate 样本中激活率 - benign 样本中激活率
hate_mask   = lbl_all == 1
benign_mask = lbl_all == 0
hate_rate   = (z_all[hate_mask]   > 0).mean(axis=0)   # [D]
benign_rate = (z_all[benign_mask] > 0).mean(axis=0)   # [D]
hate_bias   = hate_rate - benign_rate                  # 正值 = 偏向 hate

feat_stats = pd.DataFrame({
    'feature_id':      np.arange(D),
    'activation_rate': activation_rate,
    'mean_activation': mean_activation,
    'hate_rate':       hate_rate,
    'benign_rate':     benign_rate,
    'hate_bias':       hate_bias,
    'abs_bias':        np.abs(hate_bias),
})
feat_stats = feat_stats.sort_values('activation_rate', ascending=False)
feat_stats.to_csv(f"{OUT_DIR}/concept_report.csv", index=False)
print(f"  ✓ concept_report.csv 保存完成")

# ── 2. Top-activating sample retrieval ───────────────────────
print("\nRetrieving top-activating samples for top features...")

# 选 top-20 最常激活的 feature
top_features = feat_stats.head(20)['feature_id'].tolist()
# 再加 top-10 hate-biased features
top_hate_features = feat_stats.nlargest(10, 'hate_bias')['feature_id'].tolist()
features_to_analyze = list(set(top_features + top_hate_features))

TOP_K_SAMPLES = 5  # 每个 feature 展示 top-5 样本

lines = []
lines.append("=" * 70)
lines.append("InterVLM: Top-Activating Samples per SAE Feature")
lines.append("=" * 70)

for fid in features_to_analyze:
    acts = z_all[:, fid]
    top_idx = np.argsort(acts)[::-1][:TOP_K_SAMPLES]

    stat = feat_stats[feat_stats['feature_id'] == fid].iloc[0]
    lines.append(f"\n{'─'*60}")
    lines.append(f"Feature #{fid}")
    lines.append(f"  activation_rate: {stat['activation_rate']:.3f}")
    lines.append(f"  hate_bias:       {stat['hate_bias']:+.3f}  "
                 f"(hate={stat['hate_rate']:.3f} benign={stat['benign_rate']:.3f})")
    lines.append(f"  Top-{TOP_K_SAMPLES} activating samples:")

    for rank, idx in enumerate(top_idx):
        row   = df_all.iloc[idx]
        label = "HATE" if lbl_all[idx] == 1 else "benign"
        text  = str(row['text'])[:120].replace('\n', ' ')
        lines.append(f"  [{rank+1}] [{label}] act={acts[idx]:.3f} | {text}")

with open(f"{OUT_DIR}/top_features.txt", 'w') as f:
    f.write('\n'.join(lines))
print(f"  ✓ top_features.txt 保存完成")

# ── 3. UMAP 可视化 ────────────────────────────────────────────
try:
    import umap
    print("\nRunning UMAP...")

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)

    # 3a. 原始 h 空间 UMAP
    print("  Fitting UMAP on h (original embeddings)...")
    h_2d = reducer.fit_transform(h_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(h_2d[:, 0], h_2d[:, 1],
                         c=lbl_all, cmap='RdBu_r', alpha=0.6, s=8)
    plt.colorbar(scatter, ax=ax, label='label (1=hate)')
    ax.set_title("UMAP of MemeCLIP Fused Representation (h)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/umap_h.png", dpi=150)
    plt.close()
    print(f"  ✓ umap_h.png 保存完成")

    # 3b. SAE latent z 空间 UMAP（用激活率最高的 200 维降维）
    print("  Fitting UMAP on z (SAE latents, top-200 dims)...")
    top200_dims = feat_stats.head(200)['feature_id'].values
    z_sub = z_all[:, top200_dims]
    z_2d = reducer.fit_transform(z_sub)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1],
                         c=lbl_all, cmap='RdBu_r', alpha=0.6, s=8)
    plt.colorbar(scatter, ax=ax, label='label (1=hate)')
    ax.set_title("UMAP of SAE Latent Space (z, top-200 features)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/umap_z.png", dpi=150)
    plt.close()
    print(f"  ✓ umap_z.png 保存完成")

except ImportError:
    print("  ⚠ umap-learn 未安装，跳过 UMAP。运行: pip install umap-learn")

# ── 4. 简单摘要打印 ───────────────────────────────────────────
print("\n=== Concept Analysis 摘要 ===")
print(f"  总 feature 数:        {D}")
print(f"  激活率 > 5% 的 feature: {(activation_rate > 0.05).sum()}")
print(f"  激活率 > 1% 的 feature: {(activation_rate > 0.01).sum()}")
print(f"\n  Top-5 最常激活 feature:")
for _, row in feat_stats.head(5).iterrows():
    print(f"    feature #{int(row.feature_id):4d}  "
          f"rate={row.activation_rate:.3f}  bias={row.hate_bias:+.3f}")
print(f"\n  Top-5 hate-biased feature:")
for _, row in feat_stats.nlargest(5, 'hate_bias').iterrows():
    print(f"    feature #{int(row.feature_id):4d}  "
          f"rate={row.activation_rate:.3f}  bias={row.hate_bias:+.3f}")
print(f"\n  Top-5 benign-biased feature:")
for _, row in feat_stats.nsmallest(5, 'hate_bias').iterrows():
    print(f"    feature #{int(row.feature_id):4d}  "
          f"rate={row.activation_rate:.3f}  bias={row.hate_bias:+.3f}")
print(f"\n全部完成！输出目录: {OUT_DIR}")
