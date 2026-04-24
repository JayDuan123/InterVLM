# 🧩 InterVLM: Discovering Interpretable Multimodal Concepts in Meme Representations via Sparse Autoencoders

**InterVLM** is a post-hoc interpretability framework that applies **Sparse Autoencoders (SAEs)** to decompose fused MemeCLIP embeddings into sparse, human-interpretable multimodal features. Trained on **20,827 memes** from three public datasets (PrideMM, HMC, Memotion), InterVLM recovers **2,493 concept-aligned features** — **6.5× more** than raw MemeCLIP neurons — providing empirical evidence for the **superposition hypothesis** in multimodal representations.

### 🔍 Key Features

* 🧠 **Sparse feature decomposition** — A TopK SAE (k=64, 4,096 latents) trained on fused MemeCLIP ViT-L/14 embeddings with near-zero dead neurons
* 🎯 **Multimodal concept mapping** — Quantitative alignment across 8 standardized concept dimensions: *hate, stance, humor, offensive, sentiment, sarcasm, motivational, target*
* 🔬 **Causal feature analysis** — Zero-ablation and activation steering (×5) against linear probes reveal functionally important features per task
* 📊 **Three analysis modalities**:
  + Token-level activation panels (OCR-driven semantic patterns)
  + OCR-local vs. image-local vs. meme-global activation distributions
  + Concept alignment heatmaps with Fisher's exact test + BH correction
* 🏥 **Broad applications** — Enables:
  + Hateful meme auditing and bias detection
  + Interpretable feature steering for controlled generation
  + Cross-concept semantic analysis (stance, humor, offensive, sentiment, …)
  + Superposition analysis in vision–language models
  + ...

---

## ⚙️ Installation

### Prerequisites

* Python ≥ 3.9
* PyTorch ≥ 2.0
* CUDA ≥ 11.8 (optional, for GPU training)
* Other dependencies (see [`requirements.txt`](./requirements.txt))

Installation typically takes less than 10 minutes.

### Set up environment with Conda

```bash
conda create -n intervlm python=3.10
conda activate intervlm
```

### Install required packages

```bash
git clone https://github.com/JayDuan123/InterVLM.git
cd InterVLM
pip install -r requirements.txt
```

---

## 📈 Quickstart: Using InterVLM as a Feature Decomposer

Use the trained SAE to decompose any MemeCLIP fused embedding into sparse, interpretable features.

### 🔹 Request Pretrained SAE Weights

Pretrained SAE checkpoints (TopK, k=64, 4,096 latents, trained on the merged 20,827-meme corpus) are currently released on a **request basis** to approved academic users. To request access, please email [lli643@wisc.edu](mailto:lli643@wisc.edu) with the following information:

```
Subject: [InterVLM] Pretrained Weights Request — <Your Name>, <Institution>

Name:
Affiliation / Lab:
Position (e.g., PhD student / Postdoc / Faculty / Research engineer):
Institutional email:
Intended use (1–3 sentences):
    - Research question / application
    - Will the weights be redistributed? (yes / no)
    - Will derivative models / features be released? (yes / no)

I agree to:
    [ ] Use the weights for non-commercial academic research only
    [ ] Not redistribute the weights to third parties
    [ ] Cite the InterVLM paper in any resulting publications
    [ ] Comply with the licensing terms of PrideMM, HMC, and Memotion
```

Requests are typically processed within **3–5 business days**. Once approved, you will receive a time-limited download link and a `checksum.txt` for verifying file integrity.

> **Note.** Because the underlying training data (PrideMM, HMC, Memotion) contains sensitive and potentially harmful content (hate speech, slurs, targeted imagery), access is restricted to verified academic users and the weights must not be used to generate or amplify harmful content.

### 🔹 Load the Pretrained SAE

```python
from network.sae import TopKSAE
from network.memeclip import load_memeclip
import torch

# Initialize the MemeCLIP backbone (ViT-L/14) and SAE
memeclip = load_memeclip(device='cuda').eval()
sae = TopKSAE(d_in=768, d_hidden=4096, k=64).cuda().eval()

# Load pretrained SAE weights
checkpoint_path = 'pretrained/sae_topk64_4096.pth'
print('Loading SAE from', checkpoint_path)
sae.load_state_dict(torch.load(checkpoint_path))
```

### 🔹 Extract Sparse Features from a Meme

To decompose a meme, you need:

1. A **meme image** (any common format, e.g., PNG/JPG)
2. The corresponding **OCR-extracted text** (or run OCR on the fly via `utils.ocr`)

```python
from PIL import Image
from utils import ocr, preprocess_image

# Load meme
image = Image.open('examples/meme.png').convert('RGB')
text = ocr(image)                         # OCR-extracted caption

# Encode with MemeCLIP (L2-normalized)
with torch.no_grad():
    v = memeclip.encode_image(preprocess_image(image).cuda())  # (1, 768)
    t = memeclip.encode_text(text)                             # (1, 768)
    v = v / v.norm(dim=-1, keepdim=True)
    t = t / t.norm(dim=-1, keepdim=True)
    h = v * t                                                  # fused embedding

    # Sparse decomposition
    z = sae.encode(h)        # (1, 4096), only top-64 nonzero
    h_recon = sae.decode(z)  # reconstruction

# Inspect active features
active_idx = z[0].nonzero().squeeze().tolist()
print(f'Active SAE features: {active_idx}')
```

You can then use `z` for concept mapping, feature attribution, or downstream interpretability analyses.

### 🔹 Concept Alignment for a Feature

```python
from analysis.concept_mapping import concept_alignment

# Assess whether feature #3973 is aligned with the 'hate' concept
report = concept_alignment(
    feature_idx=3973,
    concept='hate',
    sae=sae,
    memeclip=memeclip,
    dataset='merged',   # PrideMM + HMC + Memotion
)
print(report)   # -> F1, Fisher p-value, BH-corrected q-value, top activating memes
```

---

## 🩺 Training InterVLM from Scratch

For the full training pipeline (data merging → MemeCLIP embedding extraction → SAE training → analysis):

📖 See [`tutorial/train_intervlm.ipynb`](./tutorial/train_intervlm.ipynb)

The tutorial covers:

* Merging PrideMM, HMC, and Memotion into a unified 20,827-meme corpus with 8 label dimensions
* Extracting fused MemeCLIP embeddings (`h = v̂ ⊙ t̂ ∈ ℝ⁷⁶⁸`)
* Training the TopK SAE (k=64, 4,096 latents, 50 epochs)
* Concept mapping via Fisher's exact test + BH correction (FDR < 0.05)
* Feature influence via zero-ablation and steering
* Generating all figures from the paper

---

## 📂 Repository Structure

```
InterVLM/
├── datasets/              # Dataset loaders (PrideMM, HMC, Memotion) + merging
├── network/               # MemeCLIP backbone + TopK SAE
├── pretrained/            # Pretrained SAE checkpoints
├── analysis/              # Concept mapping, ablation, steering, linear probes
├── visualization/         # Token panels, activation patterns, heatmaps
├── tutorial/              # End-to-end Jupyter notebooks
├── utils/                 # OCR, preprocessing, metrics
├── requirements.txt
└── README.md
```

---

## 📊 Main Results

| Metric | MemeCLIP neurons | Shuffled SAE | **InterVLM SAE** |
|---|---|---|---|
| Significant feature–concept alignments (F1 > 0.35) | 384 | 18 | **2,493** (6.5×) |
| Dead features | — | — | **0** |
| Reconstruction loss (L2) | — | — | **0.295** |
| Hate-task linear probe AUC | 0.674 | — | 0.674 |

Features also exhibit causal influence on downstream predictions via zero-ablation and ×5 steering across *hate*, *stance*, and *humor* tasks.

---

## 📚 Citation

If you use **InterVLM** in your research or applications, please cite:

```bibtex
@article{duan2025intervlm,
  title={InterVLM: Discovering Interpretable Multimodal Concepts in Meme Representations via Sparse Autoencoders},
  author={Duan, Jay and Li, Linyan and Chen, Nuo and Feng, Wanyi},
  journal={arXiv preprint},
  year={2025}
}
```

## 🙏 Acknowledgements

This work builds on [InterPLM](https://github.com/ElanaPearl/interPLM) (Simon & Zou, 2024) and [MemeCLIP](https://github.com/SiddhantBikram/MemeCLIP) (Shah et al., 2024). We thank the authors of PrideMM, the Hateful Memes Challenge, and Memotion for releasing their datasets.

## 📬 Contact

For questions or collaboration, please open an issue or email [yduan54@wisc.edu](mailto:yduan54@wisc.edu).
