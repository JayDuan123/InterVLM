"""
feature_influence.py  —  InterVLM Feature Influence Analysis
对每个 SAE feature 做:
  1. Zero-ablation: feature 置 0，看 linear probe 预测变化
  2. Steering: feature 放大 5x，看预测方向
输出:
  sae_outputs/feature_influence.csv    每个 feature 对每个 task 的影响
  sae_outputs/figures/fig10_influence_hate.png
  sae_outputs/figures/fig11_influence_all.png
"""

import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading data...")
tr_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
te_lat = torch.load(f"{OUT_DIR}/test_latents.pt")
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")
sae_ck = torch.load(f"{OUT_DIR}/sae_model.pt", map_location='cpu')

z_tr  = tr_lat['z'].numpy()      # [N_tr, 4096]
z_te  = te_lat['z'].numpy()      # [N_te, 4096]
h_tr  = tr_lat['embeddings'].float().numpy()
h_te  = te_lat['embeddings'].float().numpy()

# SAE decoder weights [input_dim, latent_dim]
W_dec = sae_ck['state_dict']['decoder.weight'].numpy()  # [768, 4096]
b_dec = sae_ck['state_dict']['decoder.bias'].numpy()    # [768]

# norm stats
norm  = torch.load(f"{OUT_DIR}/norm_stats.pt")
mean_h = norm['mean'].numpy(); std_h = norm['std'].numpy()

def sae_decode(z):
    """z: [N, 4096] → h_recon: [N, 768]"""
    return z @ W_dec.T + b_dec

# ── 训练 linear probes ────────────────────────────────────────
print("\nTraining linear probes on SAE latents...")

TASKS = {
    'hate':   {'n_classes': 2},
    'stance': {'n_classes': 3},
    'humour': {'n_classes': 2},
}

z_tr_norm = normalize(z_tr, norm='l2')
z_te_norm = normalize(z_te, norm='l2')

probes = {}
base_probs = {}  # baseline predictions on test set

for task, info in TASKS.items():
    y_tr = tr_emb[task].numpy()
    y_te = te_emb[task].numpy()

    # 过滤 -1（NaN）
    valid_tr = y_tr >= 0
    valid_te = y_te >= 0

    clf = LogisticRegression(max_iter=1000, C=1.0,
                              multi_class='auto', random_state=42)
    clf.fit(z_tr_norm[valid_tr], y_tr[valid_tr])

    prob = clf.predict_proba(z_te_norm[valid_te])

    if info['n_classes'] == 2:
        auc = roc_auc_score(y_te[valid_te], prob[:, 1])
    else:
        auc = roc_auc_score(y_te[valid_te], prob,
                            multi_class='ovr', average='macro')

    probes[task]     = (clf, valid_te, y_te)
    base_probs[task] = prob
    print(f"  {task:8s}  AUC={auc:.4f}  n_test={valid_te.sum()}")

# ── Feature influence via ablation + steering ─────────────────
print(f"\nAnalyzing {z_te.shape[1]} features × {len(TASKS)} tasks...")
print("  (ablation: feature=0, steering: feature×5)")

N_FEATURES = z_te.shape[1]
results = []

# 预计算 baseline probs（full precision）
base_full = {}
for task, (clf, valid_te, y_te) in probes.items():
    base_full[task] = clf.predict_proba(z_te_norm[valid_te])   # [N_valid, C]

for fid in range(N_FEATURES):
    row = {'feature_id': fid}

    for task, (clf, valid_te, y_te) in probes.items():
        # ── Ablation: set feature to 0 ──
        z_abl        = z_te.copy()
        z_abl[:, fid] = 0.0
        z_abl_norm   = normalize(z_abl, norm='l2')
        prob_abl     = clf.predict_proba(z_abl_norm[valid_te])

        # ── Steering: amplify feature 5x ──
        z_steer        = z_te.copy()
        z_steer[:, fid] *= 5.0
        z_steer_norm   = normalize(z_steer, norm='l2')
        prob_steer     = clf.predict_proba(z_steer_norm[valid_te])

        base = base_full[task]

        if task in ['hate', 'humour']:
            # Binary: use class-1 probability
            delta_abl   = (prob_abl[:,1]   - base[:,1]).mean()
            delta_steer = (prob_steer[:,1] - base[:,1]).mean()
        else:
            # Multiclass stance: use max class probability change
            delta_abl   = (prob_abl   - base).mean(axis=0)
            delta_steer = (prob_steer - base).mean(axis=0)
            # summarize as max absolute change
            delta_abl   = delta_abl[np.argmax(np.abs(delta_abl))]
            delta_steer = delta_steer[np.argmax(np.abs(delta_steer))]

        row[f'{task}_delta_ablation'] = float(delta_abl)
        row[f'{task}_delta_steering'] = float(delta_steer)

    results.append(row)

    if fid % 500 == 0:
        print(f"  processed {fid}/{N_FEATURES}...")

df_inf = pd.DataFrame(results)
df_inf.to_csv(f"{OUT_DIR}/feature_influence.csv", index=False)
print(f"\n✓ feature_influence.csv saved")

# ── 统计最有影响力的 feature ──────────────────────────────────
print("\n=== Top influential features ===")
for task in TASKS:
    col_abl = f'{task}_delta_ablation'
    col_str = f'{task}_delta_steering'

    # hate-promoting features: steering → positive delta
    top_promote = df_inf.nlargest(5, col_str)[['feature_id', col_str, col_abl]]
    top_suppress= df_inf.nsmallest(5, col_str)[['feature_id', col_str, col_abl]]

    print(f"\n  {task} — top 5 promoting (steering ↑):")
    print(top_promote.to_string(index=False))
    print(f"  {task} — top 5 suppressing (steering ↓):")
    print(top_suppress.to_string(index=False))

# ── Fig 10: Hate influence scatter ───────────────────────────
print("\nPlotting figures...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, task in zip(axes, TASKS):
    col_abl = f'{task}_delta_ablation'
    col_str = f'{task}_delta_steering'
    x = df_inf[col_abl].values
    y = df_inf[col_str].values

    # color by influence magnitude
    mag = np.sqrt(x**2 + y**2)
    sc  = ax.scatter(x, y, c=mag, cmap='YlOrRd',
                     s=4, alpha=.5, linewidths=0)
    plt.colorbar(sc, ax=ax, shrink=.7, label='|influence|')

    # highlight top-10 most influential
    top10 = np.argsort(mag)[::-1][:10]
    ax.scatter(x[top10], y[top10],
               c='black', s=40, alpha=.9, linewidths=0, zorder=5)
    for idx in top10:
        ax.annotate(f"#{idx}", (x[idx], y[idx]),
                    fontsize=7, xytext=(3,3),
                    textcoords='offset points', color='#333')

    ax.axhline(0, color='gray', lw=.6, ls='--', alpha=.5)
    ax.axvline(0, color='gray', lw=.6, ls='--', alpha=.5)
    ax.set_xlabel('Δ prob (ablation)', fontsize=9)
    ax.set_ylabel('Δ prob (steering ×5)', fontsize=9)
    ax.set_title(f'{task}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=.1)

plt.suptitle('InterVLM: Feature Influence (Ablation vs Steering)\n'
             'each point = one SAE feature; black = top-10 most influential',
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig10_feature_influence.png", dpi=150)
plt.close()
print("✓ fig10_feature_influence.png")

# ── Fig 11: Top-20 influential features per task ─────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 8))
for ax, task in zip(axes, TASKS):
    col_str = f'{task}_delta_steering'
    top10_pos = df_inf.nlargest(10, col_str)
    top10_neg = df_inf.nsmallest(10, col_str)
    combined  = pd.concat([top10_pos, top10_neg])
    combined  = combined.sort_values(col_str)

    colors = ['#E24B4A' if v > 0 else '#378ADD'
              for v in combined[col_str]]
    bars = ax.barh(
        [f"#{int(r)}" for r in combined['feature_id']],
        combined[col_str],
        color=colors, alpha=.8, edgecolor='none'
    )
    ax.bar_label(bars, fmt='%+.4f', padding=3, fontsize=7)
    ax.axvline(0, color='gray', lw=.8, ls='--')
    ax.set_title(f'{task}\n(red=promoting, blue=suppressing)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Δ prob (steering ×5)', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=.15)

plt.suptitle('InterVLM: Top-20 Feature Influence per Task',
             fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_top_influence_per_task.png", dpi=150)
plt.close()
print("✓ fig11_top_influence_per_task.png")
print("\n完成！")
