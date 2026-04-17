"""
token_activation.py  —  InterVLM Token-level Feature Activation
对每个代表性 SAE feature，生成：
  左：OCR token 贡献 bar chart（类似 InterPLM Figure 1c/d）
  右：meme 图片，贡献最大的词高亮标注
每个 feature × top-6 样本 = 一张 panel
"""

import os, sys, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from textwrap import wrap

sys.path.insert(0, "/projectnb/cepinet/users/Jay/InterVLP/MemeCLIP/code")
from clip import clip

BASE    = "/projectnb/cepinet/users/Jay/InterVLP"
OUT_DIR = f"{BASE}/sae_outputs"
DATA    = f"{BASE}/MemeCLIP/data"
FIG_DIR = f"{OUT_DIR}/figures/token_panels"
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading data...")
tr_lat = torch.load(f"{OUT_DIR}/train_latents.pt")
te_lat = torch.load(f"{OUT_DIR}/test_latents.pt")
tr_emb = torch.load(f"{OUT_DIR}/train_embeddings.pt")
te_emb = torch.load(f"{OUT_DIR}/test_embeddings.pt")
sae_ck = torch.load(f"{OUT_DIR}/sae_model.pt", map_location='cpu')

z_all = np.concatenate([tr_lat['z'].numpy(), te_lat['z'].numpy()])

def get_label(key, na=-1):
    trl = tr_emb[key].numpy() if key in tr_emb else np.full(len(tr_emb['embeddings']), na)
    tel = te_emb[key].numpy() if key in te_emb else np.full(len(te_emb['embeddings']), na)
    return np.concatenate([trl, tel])

hate   = get_label('hate')
stance = get_label('stance')
humor  = get_label('humor')

df = pd.concat([
    pd.read_csv(f"{DATA}/merged_train.csv"),
    pd.read_csv(f"{DATA}/merged_test.csv"),
], ignore_index=True).iloc[:z_all.shape[0]].reset_index(drop=True)

# SAE decoder weights: [input_dim, latent_dim]
W_dec = sae_ck['state_dict']['decoder.weight'].numpy()  # [768, 4096]

# ── 加载 CLIP ─────────────────────────────────────────────────
print("Loading CLIP...")
clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE, jit=False)
clip_model.float().eval()

# ── Token contribution 计算 ───────────────────────────────────
@torch.no_grad()
def token_contributions(text, feature_id):
    """
    对文本里每个 token，算它对 feature_id 的贡献
    方法：encode 每个 token → 和 SAE decoder 的 feature direction 做点积
    返回 (tokens_list, contributions_array)
    """
    # 分词（CLIP tokenizer）
    tokens_raw = clip.tokenize([text], context_length=77, truncate=True)
    token_ids  = tokens_raw[0].numpy()

    # 找到非零 token 的位置（去掉 padding）
    # CLIP token: 49406=SOT, 49407=EOT, 0=PAD
    valid_pos = []
    for i, tid in enumerate(token_ids):
        if tid == 49406: continue  # SOT
        if tid == 49407: break     # EOT
        if tid == 0:     continue  # PAD
        valid_pos.append(i)

    if len(valid_pos) == 0:
        return [], np.array([])

    # 对每个 token 单独 encode，算 contribution
    # 近似：把每个 token 的 embedding 投影到 feature decoder direction
    feature_dir = W_dec[:, feature_id]                  # [768]
    feature_dir = feature_dir / (np.linalg.norm(feature_dir) + 1e-8)

    # 用 CLIP 的 token embedding（不过 transformer，直接用词向量）
    token_embeds = clip_model.token_embedding(
        tokens_raw.to(DEVICE)
    ).squeeze(0).cpu().numpy()   # [77, 512] for ViT-L/14 it's 768

    # 如果 token_embed dim 和 feature_dir dim 不同，跳过
    if token_embeds.shape[1] != len(feature_dir):
        # 用全句 encode 的替代方法
        return _fallback_contributions(text, feature_id, valid_pos, token_ids)

    contribs = []
    token_strs = []

    # decode token id 到字符串
    vocab = clip_model.vocab if hasattr(clip_model, 'vocab') else None

    for pos in valid_pos:
        emb    = token_embeds[pos]                         # [768]
        emb_n  = emb / (np.linalg.norm(emb) + 1e-8)
        contrib = float(np.dot(emb_n, feature_dir))
        contribs.append(contrib)

        # 尝试 decode token
        tid = token_ids[pos]
        try:
            tok_str = clip._tokenizer.decoder.get(tid, f"<{tid}>")
            tok_str = tok_str.replace('</w>', '').strip()
        except Exception:
            tok_str = f"tok{pos}"
        token_strs.append(tok_str if tok_str else f"t{pos}")

    return token_strs, np.array(contribs)


def _fallback_contributions(text, feature_id, valid_pos, token_ids):
    """fallback: 对每个词单独 encode 整句"""
    words = text.split()[:20]
    feature_dir = W_dec[:, feature_id]
    feature_dir = feature_dir / (np.linalg.norm(feature_dir) + 1e-8)
    contribs = []
    for w in words:
        tok = clip.tokenize([w], context_length=77, truncate=True).to(DEVICE)
        feat = clip_model.encode_text(tok).float().cpu().numpy()[0]
        feat_n = feat / (np.linalg.norm(feat) + 1e-8)
        contribs.append(float(np.dot(feat_n, feature_dir)))
    return words, np.array(contribs)


# ── 代表性 feature ────────────────────────────────────────────
FEATURES = {
    2105: ('Stance = Support',            '+0.072', '#2ECC71'),
    2405: ('Stance = Oppose',             '-0.072', '#E74C3C'),
    1299: ('Benign-biased',               '-0.071', '#378ADD'),
    2011: ('Humor-biased',                '-0.054', '#9B59B6'),
}

# ── Panel 生成 ────────────────────────────────────────────────
def label_str(idx):
    h = int(hate[idx]); s = int(stance[idx])
    parts = []
    if h == 1: parts.append('HATE')
    elif h == 0: parts.append('benign')
    if s == 1: parts.append('support')
    elif s == 2: parts.append('oppose')
    return ' | '.join(parts) if parts else 'N/A'

def label_color(idx):
    h = int(hate[idx]); s = int(stance[idx])
    if h == 1: return '#E24B4A'
    if s == 2: return '#E74C3C'
    if s == 1: return '#2ECC71'
    return '#378ADD'

def make_token_panel(fid, concept_name, bias_str, concept_color, top_k=6):
    acts    = z_all[:, fid]
    top_idx = np.argsort(acts)[::-1][:top_k]
    act_rate = (acts > 0).mean()

    fig, axes = plt.subplots(top_k, 2,
                             figsize=(14, top_k * 3.2 + 1.2),
                             gridspec_kw={'width_ratios': [2, 1]})

    fig.suptitle(
        f"Feature #{fid}  —  {concept_name}   "
        f"bias={bias_str}   act_rate={act_rate:.3f}",
        fontsize=13, fontweight='bold', color=concept_color, y=0.995
    )

    for rank, idx in enumerate(top_idx):
        row      = df.iloc[idx]
        text     = str(row.get('text', ''))
        lc       = label_color(idx)
        ls       = label_str(idx)
        act_val  = acts[idx]

        # 左图：token contribution bar chart
        ax_bar = axes[rank, 0]
        tokens, contribs = token_contributions(text, fid)

        if len(tokens) > 0 and len(contribs) > 0:
            # 归一化到 [0,1]
            c_min, c_max = contribs.min(), contribs.max()
            c_norm = (contribs - c_min) / (c_max - c_min + 1e-8)

            x_pos  = np.arange(len(tokens))
            colors_bar = [concept_color if v > 0.5 else '#cccccc' for v in c_norm]
            ax_bar.bar(x_pos, c_norm, color=colors_bar, alpha=0.8, width=0.7)

            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax_bar.set_ylabel('Feature activation\ncontribution', fontsize=8)
            ax_bar.set_ylim(0, 1.15)
            ax_bar.axhline(0.5, color='gray', lw=0.7, ls='--', alpha=0.5)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)

            # 标出贡献最大的词
            top_tok_idx = np.argmax(c_norm)
            ax_bar.annotate(
                f"↑ {tokens[top_tok_idx]}",
                (top_tok_idx, c_norm[top_tok_idx] + 0.05),
                fontsize=8, color=concept_color, fontweight='bold',
                ha='center'
            )
        else:
            ax_bar.text(0.5, 0.5, 'no tokens', ha='center', va='center',
                        transform=ax_bar.transAxes, color='#999')

        # 标题：标签 + activation
        ax_bar.set_title(
            f"[{ls}]  act={act_val:.3f}   sample #{idx}",
            fontsize=8.5, color=lc, loc='left', pad=3
        )

        # 右图：meme 图片
        ax_img = axes[rank, 1]
        img_path = str(row.get('image', ''))
        loaded   = False
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((300, 300))
                ax_img.imshow(img)
                loaded = True
            except Exception:
                pass
        if not loaded:
            ax_img.set_facecolor('#f5f5f5')
            ax_img.text(0.5, 0.5, '[N/A]', ha='center', va='center',
                        fontsize=9, color='#aaa', transform=ax_img.transAxes)
        ax_img.axis('off')
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(lc)
            spine.set_linewidth(1.8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out = f"{FIG_DIR}/token_panel_f{fid}.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ token_panel_f{fid}.png  [{concept_name}]")


print("\nGenerating token activation panels...")
for fid, (concept, bias_str, color) in FEATURES.items():
    make_token_panel(fid, concept, bias_str, color, top_k=6)

print(f"\n完成！保存在: {FIG_DIR}")
