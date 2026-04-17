"""
extract_clip_embeddings.py
从合并数据集提取 CLIP fused embedding，保存所有标签维度
"""
import os, torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from clip import clip

BASE     = "/projectnb/cepinet/users/Jay/InterVLP"
DATA_DIR = f"{BASE}/MemeCLIP/data"
OUT_DIR  = f"{BASE}/sae_outputs"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading CLIP ViT-L/14 on {DEVICE}...")
clip_model, preprocess = clip.load("ViT-L/14", device=DEVICE, jit=False)
clip_model.float().eval()
print("✓ CLIP loaded")

@torch.no_grad()
def extract(image_path, text):
    try:
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    except Exception:
        return None
    tokens    = clip.tokenize([str(text)], context_length=77, truncate=True).to(DEVICE)
    img_feat  = clip_model.encode_image(img).float()
    text_feat = clip_model.encode_text(tokens).float()
    img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return (img_feat * text_feat).squeeze(0).cpu()

# 标签列及其 NA 值
LABEL_COLS = {
    'hate':        -1,
    'target':      -1,
    'stance':      -1,
    'humor':       -1,
    'offensive':   -1,
    'sentiment':   -99,
    'motivational':-1,
    'sarcasm':     -1,
}

for split in ["train", "test"]:
    # 优先用 merged，否则用原始
    merged = f"{DATA_DIR}/merged_{split}.csv"
    orig   = f"{DATA_DIR}/{split}.csv"
    csv_path = merged if os.path.exists(merged) else orig

    if not os.path.exists(csv_path):
        print(f"⚠ 找不到 {csv_path}，跳过")
        continue

    df = pd.read_csv(csv_path)
    print(f"\n处理 {split} ({os.path.basename(csv_path)}): {len(df)} 条...")

    embeddings = []
    label_lists = {col: [] for col in LABEL_COLS}
    sources    = []
    valid_idx  = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        h = extract(row['image'], row['text'])
        if h is None:
            continue
        embeddings.append(h)
        for col, na_val in LABEL_COLS.items():
            val = row.get(col, na_val)
            try:
                val = int(val) if not pd.isna(val) else na_val
            except:
                val = na_val
            label_lists[col].append(val)
        sources.append(str(row.get('source', 'unknown')))
        valid_idx.append(i)

    emb_tensor = torch.stack(embeddings)
    save_dict  = {
        'embeddings': emb_tensor,
        'labels':     torch.tensor(label_lists['hate']),  # 向后兼容
        'sources':    sources,
        'indices':    valid_idx,
    }
    for col in LABEL_COLS:
        save_dict[col] = torch.tensor(label_lists[col])

    out_path = f"{OUT_DIR}/{split}_embeddings.pt"
    torch.save(save_dict, out_path)
    print(f"✓ {split}: {emb_tensor.shape}  →  {out_path}")

    # 统计
    for col, na_val in LABEL_COLS.items():
        t = save_dict[col]
        n = (t != na_val).sum().item()
        print(f"  {col:15s}: {n:6d} 有效")

print("\n完成！")
