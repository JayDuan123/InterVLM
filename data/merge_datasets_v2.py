"""
merge_datasets_v2.py
合并 PrideMM + HMC + Memotion，统一标签，去除推导性重叠
标签来源：
  hate:         PrideMM + HMC（独立标注）
  offensive:    Memotion only（独立标注）
  humor:        PrideMM + Memotion（合并）
  stance:       PrideMM only
  sentiment:    Memotion only
  motivational: Memotion only
  sarcasm:      Memotion only
  target:       PrideMM only
"""

import os, json
import pandas as pd
import numpy as np
from pathlib import Path

BASE     = Path("/projectnb/cepinet/users/Jay/InterVLP")
OUT_DIR  = BASE / "MemeCLIP/data"

def memotion_humor(h):
    return {'not_funny':0,'funny':1,'very_funny':2,'hilarious':3}.get(str(h).strip(),-1)

def memotion_offensive(o):
    return {'not_offensive':0,'slight':1,'very_offensive':2,
            'hateful_offensive':3}.get(str(o).strip(),-1)

def memotion_sentiment(s):
    return {'very_negative':-2,'negative':-1,'neutral':0,
            'positive':1,'very_positive':2}.get(str(s).strip(),-99)

def memotion_motivational(m):
    return {'not_motivational':0,'motivational':1}.get(str(m).strip(),-1)

def memotion_sarcasm(s):
    return {'not_sarcastic':0,'general':0,'twisted_meaning':1,
            'very_twisted':2}.get(str(s).strip(),-1)

# ══════════════════════════════════════════════════════════════
# 1. PrideMM
# ══════════════════════════════════════════════════════════════
print("Loading PrideMM...")
rows = []
for split in ['train','test']:
    df = pd.read_csv(OUT_DIR / f"{split}.csv")
    for _, r in df.iterrows():
        humor_val = 2 if int(r['humour']) == 1 else 0
        rows.append({
            'image':        str(r['image']),
            'text':         str(r['text']),
            'source':       'pridemm',
            'split':        split,
            'hate':         int(r['hate']),
            'target':       int(r['target']) if not pd.isna(r['target']) else -1,
            'stance':       int(r['stance']),
            'humor':        humor_val,
            'offensive':    -1,       # 无独立标注
            'sentiment':    -99,
            'motivational': -1,
            'sarcasm':      -1,
        })
df_pridemm = pd.DataFrame(rows)
print(f"  {len(df_pridemm)} rows  hate={df_pridemm.hate.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════
# 2. HMC
# ══════════════════════════════════════════════════════════════
print("Loading HMC...")
HMC_IMG = BASE / "HMC/data/img"
rows = []
for split, fname in [('train','train.jsonl'),('test','dev.jsonl')]:
    for line in open(BASE / f"HMC/data/{fname}"):
        d = json.loads(line)
        if 'label' not in d: continue
        rows.append({
            'image':        str(HMC_IMG / Path(d['img']).name),
            'text':         str(d.get('text','')),
            'source':       'hmc',
            'split':        split,
            'hate':         int(d['label']),
            'target':       -1,
            'stance':       -1,
            'humor':        -1,
            'offensive':    -1,       # 无独立标注
            'sentiment':    -99,
            'motivational': -1,
            'sarcasm':      -1,
        })
df_hmc = pd.DataFrame(rows)
print(f"  {len(df_hmc)} rows  hate={df_hmc.hate.value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════
# 3. Memotion
# ══════════════════════════════════════════════════════════════
print("Loading Memotion...")
MEMO_DIR = BASE / "Memotion/memotion_dataset_7k"
MEMO_IMG = MEMO_DIR / "images"

df_m = pd.read_csv(MEMO_DIR / "labels.csv")
rows = []
for _, r in df_m.iterrows():
    img_name = str(r.get('image_name', r.get('Image_name',''))).strip()
    img_path = str(MEMO_IMG / img_name)
    text     = str(r.get('text_corrected', r.get('text_ocr',''))).strip()
    hum_val  = memotion_humor(r.get('humour',''))
    off_val  = memotion_offensive(r.get('offensive',''))
    rows.append({
        'image':        img_path,
        'text':         text,
        'source':       'memotion',
        'split':        'train',
        'hate':         -1,           # 无独立标注
        'target':       -1,
        'stance':       -1,
        'humor':        hum_val,
        'offensive':    off_val,
        'sentiment':    memotion_sentiment(r.get('overall_sentiment','')),
        'motivational': memotion_motivational(r.get('motivational','')),
        'sarcasm':      memotion_sarcasm(r.get('sarcasm','')),
    })
df_memotion = pd.DataFrame(rows)
print(f"  {len(df_memotion)} rows")

# ══════════════════════════════════════════════════════════════
# 合并
# ══════════════════════════════════════════════════════════════
df_all = pd.concat([df_pridemm, df_hmc, df_memotion], ignore_index=True)
print(f"\n合并后: {len(df_all)} rows")
print(f"来源: {df_all['source'].value_counts().to_dict()}")

df_train = df_all[df_all.split=='train'].reset_index(drop=True)
df_test  = df_all[df_all.split=='test'].reset_index(drop=True)

df_train.to_csv(OUT_DIR / "merged_train.csv", index=False)
df_test.to_csv( OUT_DIR / "merged_test.csv",  index=False)

print(f"\n✓ merged_train.csv: {len(df_train)}")
print(f"✓ merged_test.csv:  {len(df_test)}")

# ══════════════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════════════
print("\n=== 各标签有效样本数（去除推导性重叠后）===")
cols = ['hate','target','stance','humor','offensive','sentiment','motivational','sarcasm']
for col in cols:
    na = -99 if col == 'sentiment' else -1
    n  = (df_all[col] != na).sum()
    src = df_all[df_all[col] != na]['source'].value_counts().to_dict()
    print(f"  {col:15s}: {n:6d} 条  来源={src}")

print("\n=== Humor 强度分布 ===")
print(df_all[df_all.humor>=0]['humor'].value_counts().sort_index().to_dict())
print("\n=== Offensive 强度分布 ===")
print(df_all[df_all.offensive>=0]['offensive'].value_counts().sort_index().to_dict())
