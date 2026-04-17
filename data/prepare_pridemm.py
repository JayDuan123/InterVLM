"""
把 PrideMM.csv 转成 MemeCLIP 格式，按 split 拆成 train/val/test CSV
输出列: image, text, label  (label = hate 字段)
"""
import pandas as pd
from pathlib import Path

BASE      = Path("/projectnb/cepinet/users/Jay/InterVLP")
CSV_PATH  = BASE / "PrideMM/PrideMM/PrideMM.csv"
IMG_DIR   = BASE / "PrideMM/PrideMM/Images"
OUT_DIR   = BASE / "MemeCLIP/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"总样本数: {len(df)}")
print(f"split 分布:\n{df['split'].value_counts()}\n")
print(f"hate 分布:\n{df['hate'].value_counts()}\n")

# 检查是否有缺失 text
n_missing = df['text'].isna().sum()
if n_missing:
    print(f"⚠ 有 {n_missing} 条 text 为空，已填充空字符串")
    df['text'] = df['text'].fillna("")

# 构建完整图片路径
df['image'] = df['name'].apply(lambda x: str(IMG_DIR / x))

# 验证图片是否存在（抽查前 5 条）
missing_imgs = [p for p in df['image'][:5] if not Path(p).exists()]
if missing_imgs:
    print(f"⚠ 以下图片不存在，请检查路径:\n{missing_imgs}")
else:
    print("✓ 图片路径验证通过（前5条）")

# 只保留 MemeCLIP 需要的列
out_cols = ['image', 'text', 'label']
df = df.rename(columns={'hate': 'label'})[out_cols]

# 按 split 拆分并保存
# PrideMM 原始 split 字段值: train / test (没有 val 则从 train 里切 10%)
splits = pd.read_csv(CSV_PATH)['split'].values

for split_name, split_key in [('train', 'train'), ('test', 'test')]:
    mask = splits == split_key
    subset = df[mask].reset_index(drop=True)
    if len(subset) == 0:
        print(f"⚠ split='{split_key}' 没有数据，跳过")
        continue
    out_path = OUT_DIR / f"{split_name}.csv"
    subset.to_csv(out_path, index=False)
    print(f"✓ {split_name}.csv  →  {len(subset)} 条  →  {out_path}")

# 如果没有 val split，从 train 里切 10% 出来
if 'val' not in set(splits):
    train_df = pd.read_csv(OUT_DIR / "train.csv")
    val_df   = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR  / "val.csv",   index=False)
    print(f"✓ 从 train 切出 val: train={len(train_df)}, val={len(val_df)}")

print("\n完成！输出文件:")
for f in sorted(OUT_DIR.glob("*.csv")):
    print(f"  {f}  ({sum(1 for _ in open(f))-1} 条)")
