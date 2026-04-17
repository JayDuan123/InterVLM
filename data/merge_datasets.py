"""
merge_datasets.py
合并 PrideMM + HMC + Memotion 成统一格式
统一标签空间（没有的标签填 -1）：
  hate, target, stance, humour,
  sentiment, humor_type, offensive, motivational, sarcasm
输出: MemeCLIP/data/merged_train.csv
      MemeCLIP/data/merged_test.csv
"""

import os, json
import pandas as pd
import numpy as np
from pathlib import Path

BASE     = Path("/projectnb/cepinet/users/Jay/InterVLP")
OUT_DIR  = BASE / "MemeCLIP/data"

# ── 标签映射函数 ──────────────────────────────────────────────

def harmfulness_to_hate(labels):
    """HarMeme labels → binary hate"""
    for l in labels:
        if 'very harmful' in l:    return 1
        if 'somewhat harmful' in l: return 1
        if 'not harmful' in l:      return 0
    return -1

def harmfulness_to_target(labels):
    """HarMeme target labels"""
    mapping = {'individual': 1, 'community': 2, 'organization': 3, 'society': 2}
    for l in labels:
        if l in mapping: return mapping[l]
    return 0  # undirected

def memotion_sentiment(s):
    mapping = {'very_positive': 2, 'positive': 1, 'neutral': 0,
               'negative': -1, 'very_negative': -2}
    return mapping.get(str(s).strip(), 0)

def memotion_humour(h):
    mapping = {'not_funny': 0, 'funny': 1, 'very_funny': 2, 'hilarious': 3}
    return mapping.get(str(h).strip(), -1)

def memotion_offensive(o):
    mapping = {'not_offensive': 0, 'slight': 1, 'very_offensive': 2,
               'hateful_offensive': 3}
    return mapping.get(str(o).strip(), -1)

def memotion_motivational(m):
    return 1 if str(m).strip() == 'motivational' else 0

def memotion_sarcasm(s):
    mapping = {'not_sarcastic': 0, 'general': 0, 'twisted_meaning': 1,
               'very_twisted': 2}
    return mapping.get(str(s).strip(), -1)

# ── 1. PrideMM ────────────────────────────────────────────────
print("Loading PrideMM...")
rows = []
for split in ['train', 'test']:
    df = pd.read_csv(OUT_DIR / f"{split}.csv")
    for _, r in df.iterrows():
        rows.append({
            'image':       str(r['image']),
            'text':        str(r['text']),
            'source':      'pridemm',
            'split':       split,
            # PrideMM labels
            'hate':        int(r['hate']),
            'target':      int(r['target']) if not pd.isna(r['target']) else -1,
            'stance':      int(r['stance']),
            'humour':      int(r['humour']),
            # Memotion labels (N/A)
            'sentiment':   -99,
            'humor_type':  -99,
            'offensive':   -99,
            'motivational':-99,
            'sarcasm':     -99,
        })
df_pridemm = pd.DataFrame(rows)
print(f"  PrideMM: {len(df_pridemm)} rows")

# ── 2. HMC ────────────────────────────────────────────────────
print("Loading HMC...")
HMC_IMG = BASE / "HMC/data/img"
rows = []
for split, fname in [('train','train.jsonl'),('test','dev.jsonl')]:
    path = BASE / f"HMC/data/{fname}"
    for line in open(path):
        d = json.loads(line)
        if 'label' not in d: continue
        rows.append({
            'image':       str(HMC_IMG / Path(d['img']).name),
            'text':        str(d.get('text','')),
            'source':      'hmc',
            'split':       split,
            'hate':        int(d['label']),
            'target':      -1,
            'stance':      -1,
            'humour':      -1,
            'sentiment':   -99,
            'humor_type':  -99,
            'offensive':   -99,
            'motivational':-99,
            'sarcasm':     -99,
        })
df_hmc = pd.DataFrame(rows)
print(f"  HMC: {len(df_hmc)} rows")

# ── 3. Memotion ───────────────────────────────────────────────
print("Loading Memotion...")
MEMO_DIR = BASE / "Memotion/memotion_dataset_7k"
MEMO_IMG = MEMO_DIR / "images"

df_m = pd.read_csv(MEMO_DIR / "labels.csv")
print(f"  Memotion columns: {list(df_m.columns)}")

rows = []
for _, r in df_m.iterrows():
    img_name = str(r.get('image_name', r.get('Image_name', ''))).strip()
    # 找到实际图片路径（jpg 或 png）
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        p = MEMO_IMG / (img_name if '.' in img_name else img_name + ext)
        if p.exists():
            img_path = str(p)
            break
    if img_path is None:
        img_path = str(MEMO_IMG / img_name)  # 保留原始，提取时处理

    text = str(r.get('text_corrected', r.get('text_ocr', ''))).strip()

    rows.append({
        'image':       img_path,
        'text':        text,
        'source':      'memotion',
        'split':       'train',   # Memotion 没有官方 test split，全用于训练
        # hate/stance 不直接有，用 offensive 近似
        'hate':        1 if memotion_offensive(r.get('offensive','')) >= 2 else 0,
        'target':      -1,
        'stance':      -1,
        'humour':      1 if memotion_humour(r.get('humour','')) >= 1 else 0,
        # Memotion 原始标签
        'sentiment':   memotion_sentiment(r.get('overall_sentiment','')),
        'humor_type':  memotion_humour(r.get('humour','')),
        'offensive':   memotion_offensive(r.get('offensive','')),
        'motivational':memotion_motivational(r.get('motivational','')),
        'sarcasm':     memotion_sarcasm(r.get('sarcasm','')),
    })

df_memotion = pd.DataFrame(rows)
print(f"  Memotion: {len(df_memotion)} rows")

# ── 合并 ──────────────────────────────────────────────────────
df_all = pd.concat([df_pridemm, df_hmc, df_memotion], ignore_index=True)
print(f"\n合并后总计: {len(df_all)} rows")
print(f"  来源分布: {df_all['source'].value_counts().to_dict()}")

# 按 split 拆分
df_train = df_all[df_all['split']=='train'].reset_index(drop=True)
df_test  = df_all[df_all['split']=='test'].reset_index(drop=True)

df_train.to_csv(OUT_DIR / "merged_train.csv", index=False)
df_test.to_csv( OUT_DIR / "merged_test.csv",  index=False)

print(f"\n✓ merged_train.csv: {len(df_train)} rows")
print(f"✓ merged_test.csv:  {len(df_test)} rows")

# ── 标签统计 ──────────────────────────────────────────────────
print("\n=== 各 concept 有效样本数 ===")
for col in ['hate','target','stance','humour','sentiment','humor_type','offensive','motivational','sarcasm']:
    valid = (df_all[col] >= 0).sum()
    print(f"  {col:15s}: {valid:6d} 条有效标签")
