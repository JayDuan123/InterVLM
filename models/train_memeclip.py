"""
train_memeclip.py
在 PrideMM 上从头训练 MemeCLIP，保存 checkpoint
直接运行: python train_memeclip.py
"""
import sys
sys.path.insert(0, "/projectnb/cepinet/users/Jay/InterVLP/MemeCLIP/code")

import os
import torch
from yacs.config import CfgNode
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from datasets import Custom_Collator, load_dataset
from MemeCLIP import create_model, MemeCLIP

# ── 路径 ──────────────────────────────────────────────────────
BASE        = "/projectnb/cepinet/users/Jay/InterVLP"
IMG_FOLDER  = f"{BASE}/PrideMM/PrideMM/Images"
INFO_FILE   = f"{BASE}/PrideMM/PrideMM/PrideMM.csv"
CKPT_PATH   = f"{BASE}/MemeCLIP/checkpoints"
CKPT_FILE   = f"{CKPT_PATH}/model.ckpt"
os.makedirs(CKPT_PATH, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
cfg = CfgNode()
cfg.root_dir    = BASE
cfg.img_folder  = IMG_FOLDER
cfg.info_file   = INFO_FILE
cfg.checkpoint_path = CKPT_PATH
cfg.checkpoint_file = CKPT_FILE

cfg.clip_variant = "ViT-L/14"
cfg.dataset_name = "Pride"
cfg.label        = "hate"
cfg.seed         = 42
cfg.test_only    = False
cfg.reproduce    = False
cfg.device       = "cuda"
cfg.gpus         = [0]

cfg.class_names  = ["Benign Meme", "Harmful Meme"]
cfg.num_classes  = 2

cfg.batch_size          = 16
cfg.image_size          = 224
cfg.num_mapping_layers  = 1
cfg.unmapped_dim        = 768
cfg.map_dim             = 1024
cfg.num_pre_output_layers = 1
cfg.drop_probs          = [0.1, 0.4, 0.2]
cfg.lr                  = 1e-4
cfg.max_epochs          = 15
cfg.ratio               = 0.2
cfg.weight_decay        = 1e-4
cfg.scale               = 30
cfg.print_model         = False

# ── 需要在 MemeCLIP.py 里加 map_dim ──────────────────────────
# 已经 patch 过了，确认一下
import MemeCLIP as mc_module
import inspect
src = inspect.getsource(mc_module.MemeCLIP.__init__)
if "self.map_dim" not in src:
    raise RuntimeError("请先 patch MemeCLIP.py：在 self.clip_model.float() 后加 self.map_dim = cfg.map_dim")

# ── 数据集 ────────────────────────────────────────────────────
seed_everything(cfg.seed, workers=True)

# datasets.py 按 split 列读取，PrideMM.csv 里只有 train/test
# 我们把 test 当 val 用（main.py 里 val_loader 和 test_loader 都指向 test set）
dataset_train = load_dataset(cfg=cfg, split='train')
dataset_test  = load_dataset(cfg=cfg, split='test')
print(f"Train: {len(dataset_train)}  Test: {len(dataset_test)}")

collator    = Custom_Collator(cfg)
train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size,
                          shuffle=True, collate_fn=collator, num_workers=0)
test_loader  = DataLoader(dataset_test,  batch_size=cfg.batch_size,
                          collate_fn=collator, num_workers=0)

# ── 训练 ──────────────────────────────────────────────────────
model = create_model(cfg)

checkpoint_callback = ModelCheckpoint(
    dirpath   = CKPT_PATH,
    filename  = 'model',
    monitor   = 'val/auroc',
    mode      = 'max',
    verbose   = True,
    save_weights_only = True,
    save_top_k = 1,
)

trainer = Trainer(
    accelerator  = 'gpu',
    devices      = cfg.gpus,
    max_epochs   = cfg.max_epochs,
    callbacks    = [checkpoint_callback],
    deterministic= False,
)

trainer.fit(model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader)

# ── 测试 ──────────────────────────────────────────────────────
print("\nLoading best checkpoint for test...")
best_model = MemeCLIP.load_from_checkpoint(
    checkpoint_path=CKPT_FILE, cfg=cfg)
trainer.test(best_model, dataloaders=test_loader)
print(f"\n✓ Best checkpoint: {CKPT_FILE}")
