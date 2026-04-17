"""
train_sae.py  —  InterVLM: Sparse Autoencoder on MemeCLIP fused embeddings
输入: embeddings/train_embeddings.pt  shape [N, 1024]
输出: sae_outputs/sae_model.pt        训练好的 SAE 权重
      sae_outputs/train_latents.pt    训练集 sparse latent codes z
      sae_outputs/test_latents.pt     测试集 sparse latent codes z
"""

import os, torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── 路径 ──────────────────────────────────────────────────────
BASE     = "/projectnb/cepinet/users/Jay/InterVLP"
EMB_DIR  = f"{BASE}/sae_outputs"
OUT_DIR  = f"{BASE}/sae_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ── SAE 超参 ──────────────────────────────────────────────────
INPUT_DIM  = 768      # MemeCLIP map_dim
LATENT_DIM = 4096      # 扩展因子 4x
TOPK       = 64        # 每个样本只保留 top-k 个激活特征
LR         = 1e-3
BATCH_SIZE = 256
EPOCHS     = 50

# ── SAE 模型（TopK 激活）─────────────────────────────────────
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, k):
        super().__init__()
        self.k       = k
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def encode(self, h):
        z_pre = self.encoder(h)
        topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def forward(self, h):
        z     = self.encode(h)
        h_hat = self.decoder(z)
        return h_hat, z

    def normalize_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)

# ── 加载数据 ──────────────────────────────────────────────────
print("Loading embeddings...")
train_data = torch.load(f"{EMB_DIR}/train_embeddings.pt")
test_data  = torch.load(f"{EMB_DIR}/test_embeddings.pt")

train_emb  = train_data['embeddings'].float()   # [N_train, 1024]
train_lbl  = train_data['labels']
test_emb   = test_data['embeddings'].float()    # [N_test,  1024]
test_lbl   = test_data['labels']

print(f"  train: {train_emb.shape}  test: {test_emb.shape}")

# 归一化（对 SAE 训练稳定性有帮助）
mean = train_emb.mean(dim=0)
std  = train_emb.std(dim=0).clamp(min=1e-6)
train_emb_norm = (train_emb - mean) / std
test_emb_norm  = (test_emb  - mean) / std   # 用 train 统计量

torch.save({'mean': mean, 'std': std}, f"{OUT_DIR}/norm_stats.pt")

train_loader = DataLoader(
    TensorDataset(train_emb_norm, train_lbl),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)

# ── 训练 ──────────────────────────────────────────────────────
model = SparseAutoencoder(INPUT_DIM, LATENT_DIM, TOPK).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nTraining SAE (TopK): input={INPUT_DIM}  latent={LATENT_DIM}  k={TOPK}")
print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  device={DEVICE}\n")

history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_recon = total_sparsity = total_loss = 0.0

    for h_batch, _ in train_loader:
        h_batch = h_batch.to(DEVICE)
        h_hat, z = model(h_batch)

        recon = nn.functional.mse_loss(h_hat, h_batch)
        loss  = recon

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_decoder()

        total_recon    += recon.item()
        total_sparsity += (z > 0).float().sum(dim=1).mean().item()
        total_loss     += loss.item()

    n = len(train_loader)
    avg = dict(recon=total_recon/n, sparsity=total_sparsity/n, loss=total_loss/n)
    history.append(avg)

    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            z_all   = model.encode(train_emb_norm.to(DEVICE))
            active  = (z_all > 0).float().mean(dim=0)
            n_dead  = (active == 0).sum().item()
            avg_act = (z_all > 0).float().sum(dim=1).mean().item()
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"recon={avg['recon']:.4f} | "
              f"avg_active={avg_act:.1f}/{LATENT_DIM}  "
              f"dead={n_dead}")

# ── 保存模型 ──────────────────────────────────────────────────
torch.save({
    'state_dict': model.state_dict(),
    'input_dim':  INPUT_DIM,
    'latent_dim': LATENT_DIM,
    'topk':       TOPK,
    'history':    history,
}, f"{OUT_DIR}/sae_model.pt")
print(f"\n✓ SAE 模型保存: {OUT_DIR}/sae_model.pt")

# ── 提取所有 latent codes ─────────────────────────────────────
model.eval()
with torch.no_grad():
    z_train = model.encode(train_emb_norm.to(DEVICE)).cpu()
    z_test  = model.encode(test_emb_norm.to(DEVICE)).cpu()

torch.save({'z': z_train, 'labels': train_lbl, 'embeddings': train_emb},
           f"{OUT_DIR}/train_latents.pt")
torch.save({'z': z_test,  'labels': test_lbl,  'embeddings': test_emb},
           f"{OUT_DIR}/test_latents.pt")

print(f"✓ train latents: {z_train.shape}  →  {OUT_DIR}/train_latents.pt")
print(f"✓ test  latents: {z_test.shape}   →  {OUT_DIR}/test_latents.pt")

# ── 简单统计 ──────────────────────────────────────────────────
active_rate = (z_train > 0).float().mean(dim=0)
print(f"\n=== SAE 统计 ===")
print(f"  latent dim:       {LATENT_DIM}")
print(f"  dead features:    {(active_rate == 0).sum().item()}")
print(f"  avg active/sample:{(z_train > 0).float().sum(dim=1).mean().item():.1f}")
print(f"  final recon loss: {history[-1]['recon']:.4f}")
