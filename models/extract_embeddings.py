"""
extract_embeddings.py
从 MemeCLIP 提取 fused multimodal representation (h)
h = element-wise product of normalized image & text features, shape [N, 1024]
保存到: /projectnb/cepinet/users/Jay/InterVLP/embeddings/
"""
import sys, os
sys.path.insert(0, "/projectnb/cepinet/users/Jay/InterVLP/MemeCLIP/code")

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from clip import clip

# ── 路径配置 ─────────────────────────────────────────────────
BASE        = "/projectnb/cepinet/users/Jay/InterVLP"
DATA_DIR    = f"{BASE}/MemeCLIP/data"
CKPT_PATH   = f"{BASE}/MemeCLIP/checkpoints/model.ckpt"  # 预训练权重路径
OUT_DIR     = f"{BASE}/embeddings"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载 configs ──────────────────────────────────────────────
from configs import cfg
cfg.device  = DEVICE
cfg.gpus    = [0]
cfg.defrost()
cfg.map_dim      = 1024
cfg.unmapped_dim = 768
cfg.num_mapping_layers   = 1
cfg.drop_probs   = [0.1, 0.4, 0.2]
cfg.num_pre_output_layers = 1
cfg.ratio        = 0.2
cfg.clip_variant = "ViT-L/14"
cfg.class_names  = ['Benign Meme', 'Harmful Meme']
cfg.num_classes  = 2
cfg.scale        = 30
cfg.freeze()

# ── 加载 CLIP ─────────────────────────────────────────────────
print(f"Loading CLIP {cfg.clip_variant} on {DEVICE}...")
clip_model, preprocess = clip.load(cfg.clip_variant, device=DEVICE, jit=False)
clip_model.float()
clip_model.eval()

# ── 加载 MemeCLIP ─────────────────────────────────────────────
from MemeCLIP import MemeCLIP

print("Loading MemeCLIP...")
model = MemeCLIP(cfg)

if os.path.exists(CKPT_PATH):
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    # Lightning checkpoint 的 state_dict 在 'state_dict' key 里
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("✓ Checkpoint loaded")
else:
    print("⚠ 未找到 checkpoint，使用随机初始化权重（仅用于调试）")

model = model.to(DEVICE)
model.eval()

# ── 提取函数：单条样本的 fused feature ───────────────────────
@torch.no_grad()
def extract_fused(image_path, text):
    # 1. CLIP image embedding
    try:
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"  ⚠ 图片读取失败 {image_path}: {e}")
        return None

    image_raw = clip_model.encode_image(img).float()       # [1, 768]

    # 2. CLIP text embedding
    tokens = clip.tokenize([str(text)], context_length=77, truncate=True).to(DEVICE)
    text_raw = model.text_encoder(tokens).float()          # [1, 768]

    # 3. Projection + Adapter（复现 common_step）
    img_proj  = model.image_map(image_raw)                 # [1, 1024]
    txt_proj  = model.text_map(text_raw)                   # [1, 1024]

    img_feat  = model.img_adapter(img_proj)                # [1, 1024]
    txt_feat  = model.text_adapter(txt_proj)               # [1, 1024]

    img_feat  = cfg.ratio * img_feat + (1 - cfg.ratio) * img_proj
    txt_feat  = cfg.ratio * txt_feat + (1 - cfg.ratio) * txt_proj

    img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    # 4. Fused representation h = element-wise product
    h = torch.mul(img_feat, txt_feat)                      # [1, 1024]
    return h.squeeze(0).cpu()

# ── 遍历所有 split ────────────────────────────────────────────
for split in ["train", "test"]:
    csv_path = f"{DATA_DIR}/{split}.csv"
    if not os.path.exists(csv_path):
        print(f"⚠ 找不到 {csv_path}，跳过")
        continue

    df = pd.read_csv(csv_path)
    print(f"\n处理 {split}.csv ({len(df)} 条)...")

    embeddings = []
    labels     = []
    valid_idx  = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        h = extract_fused(row['image'], row['text'])
        if h is not None:
            embeddings.append(h)
            labels.append(int(row['label']))
            valid_idx.append(i)

    emb_tensor = torch.stack(embeddings)          # [N, 1024]
    lbl_tensor = torch.tensor(labels)             # [N]

    out_path = f"{OUT_DIR}/{split}_embeddings.pt"
    torch.save({
        'embeddings': emb_tensor,   # shape [N, 1024]
        'labels':     lbl_tensor,   # shape [N]
        'indices':    valid_idx,     # 对应 CSV 行号
    }, out_path)

    print(f"✓ {split}: {emb_tensor.shape}  →  {out_path}")

print("\n全部完成！")
print(f"embeddings 维度: [N, 1024]，后续直接送入 SAE 训练")
