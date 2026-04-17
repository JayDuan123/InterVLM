"""
top_activating_panel.py
为代表性 SAE feature 生成 top-9 激活样本的 panel 图
每个 feature 一张图，展示：图片 + OCR 文本 + activation 值 + 标签
"""

import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from textwrap import wrap

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
DATA    = f"{BASE}/MemeCLIP/data"
FIG_DIR = f"{OUT_DIR}/figures/panels"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading data...")
tr_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
te_lat = torch.load(f"{OUT_DIR}/test_latents.pt")
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")

z_all  = np.concatenate([tr_lat['z'].numpy(), te_lat['z'].numpy()])  # [N, 4096]
hate   = np.concatenate([tr_emb['hate'].numpy(), te_emb['hate'].numpy()])

# 读 CSV，对齐到 z（取前 z.shape[0] 行）
df = pd.concat([
    pd.read_csv(f"{DATA}/merged_train.csv"),
    pd.read_csv(f"{DATA}/merged_test.csv"),
], ignore_index=True).iloc[:z_all.shape[0]].reset_index(drop=True)

print(f"  z: {z_all.shape}  df: {df.shape}")

# ── 要做 panel 的 feature（从 concept analysis 结果选代表性的）────
# hate_bias 计算
hr   = (z_all[hate==1] > 0).mean(0)
br   = (z_all[hate==0] > 0).mean(0)
bias = hr - br

# 选 top hate-biased + top benign-biased + neutral 各几个
top_hate   = np.argsort(bias)[::-1][:4]
top_benign = np.argsort(bias)[:4]
top_active = np.argsort((z_all>0).mean(0))[::-1][:2]
FEATURE_IDS = list(dict.fromkeys(
    list(top_hate) + list(top_benign) + list(top_active)
))[:10]

print(f"\nFeatures to panel: {FEATURE_IDS}")
for fid in FEATURE_IDS:
    print(f"  #{fid}  bias={bias[fid]:+.4f}  act_rate={(z_all[:,fid]>0).mean():.4f}")

# ── Panel 生成函数 ────────────────────────────────────────────
def make_panel(fid, top_k=9):
    acts     = z_all[:, fid]
    top_idx  = np.argsort(acts)[::-1][:top_k]
    b        = bias[fid]
    act_rate = (z_all[:, fid] > 0).mean()

    # feature 描述
    if b > 0.02:
        feat_type = "HATE-BIASED"
        type_color = "#E24B4A"
    elif b < -0.02:
        feat_type = "BENIGN-BIASED"
        type_color = "#378ADD"
    else:
        feat_type = "NEUTRAL"
        type_color = "#888780"

    ncols = 3
    nrows = (top_k + ncols - 1) // ncols
    fig   = plt.figure(figsize=(ncols * 5, nrows * 5.5 + 1.2))

    # 主标题
    fig.suptitle(
        f"SAE Feature #{fid}  [{feat_type}]   "
        f"hate_bias={b:+.4f}   act_rate={act_rate:.3f}",
        fontsize=13, fontweight='bold', color=type_color, y=0.98
    )

    for rank, idx in enumerate(top_idx):
        row = df.iloc[idx]
        ax  = fig.add_subplot(nrows, ncols, rank + 1)

        # 尝试加载图片
        img_loaded = False
        img_path   = str(row.get('image', ''))
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((400, 400))
                ax.imshow(img)
                img_loaded = True
            except Exception:
                pass

        if not img_loaded:
            ax.set_facecolor('#f5f5f5')
            ax.text(0.5, 0.5, '[image N/A]', ha='center', va='center',
                    fontsize=10, color='#aaa', transform=ax.transAxes)

        ax.axis('off')

        # 文本框：OCR + 标签 + activation
        hate_val  = int(row.get('hate', -1))
        label_str = "HATE" if hate_val == 1 else ("benign" if hate_val == 0 else "N/A")
        label_col = "#E24B4A" if hate_val == 1 else "#378ADD"
        act_val   = acts[idx]

        text_raw  = str(row.get('text', ''))[:200]
        text_wrap = '\n'.join(wrap(text_raw, width=38))

        # 在图片下方添加文字
        ax.set_title(
            f"[{label_str}]  act={act_val:.3f}\n{text_wrap[:120]}",
            fontsize=6.5,
            color=label_col,
            pad=3,
            loc='left'
        )

        # 彩色边框
        for spine in ax.spines.values():
            spine.set_edgecolor(label_col)
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = f"{FIG_DIR}/panel_feature_{fid}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ panel_feature_{fid}.png  [{feat_type}  bias={b:+.4f}]")

# ── 生成所有 panel ────────────────────────────────────────────
print("\nGenerating panels...")
for fid in FEATURE_IDS:
    make_panel(int(fid), top_k=9)

print(f"\n完成！所有 panel 保存在: {FIG_DIR}")
