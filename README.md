# Dynamic-Spectral-Polarity-Linear-Attention-DSP-Attention-
# DSP-Attention: Dynamic Spectral-Polarity Linear Attention

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A novel linear attention mechanism that extends **PolaFormer** (Meng et al., ICLR 2025) with spectral domain processing, dynamic context-aware weighting, multi-scale hierarchical features, and gradient-informed adaptive mixing.

---

## 📄 Base Paper

This work builds directly upon:

> **PolaFormer: Polarity-Aware Linear Attention for Vision Transformers**  
> Weikang Meng, Yadan Luo, Xin Li, Dongmei Jiang, Zheng Zhang  
> *Published at ICLR 2025*  
> 📎 Paper: https://arxiv.org/abs/2501.15061  
> 💻 Official Code: https://github.com/ZacharyMeng/PolaFormer

### What is PolaFormer?

PolaFormer addresses two key failures in existing linear attention mechanisms:

1. **Loss of negative values** — ReLU-based feature maps discard negative query-key interactions, losing crucial relational information.
2. **Loss of attention spikiness** — Without softmax's exponential scaling, linear attention produces overly uniform weight distributions (high entropy).

PolaFormer fixes this by:
- Decomposing Q/K into positive and negative components and computing **same-signed** and **opposite-signed** interactions separately
- Using a **learnable power function** (proven to reduce entropy via positive second derivative) for rescaling
- Employing **depthwise convolutions** to address the low-rank issue of the attention map

DSP-Attention preserves all of the above and introduces additional novel components on top.

---

## 🚀 Novel Contributions (DSP-Attention)

| Component | Description |
|---|---|
| **SpectralPolarityMixer** | Transforms Q and K to the frequency domain via FFT; applies learnable complex-valued filters to separate positive/negative spectral modes |
| **DynamicContextEncoder** | Encodes global, temporal (LSTM), and self-attention context to generate per-head adaptive weighting coefficients |
| **HierarchicalPolarityProcessor** | Processes polarity decomposition at multiple spatial scales (scale_factor = 1, 2, 4, …) with scale-specific learnable power functions |
| **GradientInformedMixer** | Combines multi-scale outputs using gradient-informed mixing weights with exponential moving average tracking |
| **MultiModalConvolution** | Applies multiple depthwise convolution kernels at different receptive fields and merges them with learned weights |

---

## 🏗️ Architecture Overview

```
Input x (B, N, C)
        │
        ├──────────────────────────────────┐
        │  DynamicContextEncoder           │
        │  (global + LSTM + self-attn)     │
        │         ↓ dynamic_coeffs         │
        │                                  │
        │  For each scale s in [0..S-1]:   │
        │    QKV projection                │
        │    SpectralPolarityMixer (FFT)   │
        │    HierarchicalPolarityProcessor │
        │    Same + Opposite Attn (linear) │
        │         ↓ scale_output_s         │
        │                                  │
        │  GradientInformedMixer           │
        │         ↓ combined_output        │
        │                                  │
        │  MultiModalConvolution           │
        │         ↓ enhanced_output        │
        │                                  │
        └──────► residual + LayerNorm + proj ──► Output
```

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/dsp-attention.git
cd dsp-attention
pip install torch torchvision
```

No additional dependencies beyond PyTorch are required for the core module.

---

## ⚡ Quick Start

```python
import torch
from dsp_attention import DynamicSpectralPolarityAttention, create_dsp_model

# Standalone DSP-Attention layer
attn = DynamicSpectralPolarityAttention(
    dim=384,
    num_heads=6,
    spectral_modes=16,
    num_scales=3,
)

x = torch.randn(2, 196, 384)   # (batch, tokens, dim)
out = attn(x)                   # (batch, tokens, dim)
print(out.shape)                # torch.Size([2, 196, 384])

# Full classification model
model = create_dsp_model(
    dim=384,
    num_heads=6,
    depth=12,
    num_classes=1000,
    spectral_modes=16,
    num_scales=3,
)

logits = model(x)               # (batch, num_classes)
print(logits.shape)             # torch.Size([2, 1000])
```

---

## 🔧 Configuration

### `DynamicSpectralPolarityAttention` Parameters

| Parameter | Default | Description |
|---|---|---|
| `dim` | — | Input feature dimension (must be divisible by `num_heads`) |
| `num_heads` | `8` | Number of attention heads |
| `qkv_bias` | `False` | Add learnable bias to QKV projections |
| `attn_drop` | `0.0` | Attention dropout rate |
| `proj_drop` | `0.0` | Output projection dropout rate |
| `alpha` | `4.0` | Scaling factor for learnable power function |
| `kernel_size` | `5` | Base kernel size for multi-modal convolution |
| `spectral_modes` | `16` | Number of frequency modes to retain in FFT |
| `num_scales` | `3` | Number of hierarchical polarity scales |
| `adaptive_beta` | `0.1` | Adaptation rate for dynamic weighting |

---

## 📁 File Structure

```
dsp-attention/
├── dsp_attention.py      # Core DSP-Attention implementation
├── README.md             # This file
└── LICENSE               # MIT License
```

---

## 🔬 Design Decisions

### Why FFT for polarity?
The spectral domain exposes frequency-level structure in Q and K that is invisible in the spatial domain. By applying learnable complex filters per frequency mode, DSP-Attention can learn to selectively amplify or suppress certain frequency components in the polarity decomposition.

### Why multi-scale?
Different vision tasks require attention at different spatial granularities. Hierarchical processing at multiple scales (with scale-specific power functions) allows the model to capture both local fine-grained and global coarse-grained polarity interactions.

### Why context-aware dynamic weighting?
Static mixing coefficients (like the fixed `Gs`, `Go` in PolaFormer) cannot adapt to input-dependent complexity. The `DynamicContextEncoder` uses global average pooling, bidirectional LSTM temporal encoding, and self-attention aggregation to produce input-conditioned per-head mixing coefficients.

---

## 📜 Citation

If you use this code, please cite the original PolaFormer paper:

```bibtex
@inproceedings{meng2025polaformer,
  title     = {PolaFormer: Polarity-Aware Linear Attention for Vision Transformers},
  author    = {Weikang Meng and Yadan Luo and Xin Li and Dongmei Jiang and Zheng Zhang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.15061}
}
```

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built upon the PolaFormer codebase from [ZacharyMeng/PolaFormer](https://github.com/ZacharyMeng/PolaFormer). Backbone integration patterns (PVT, Swin) referenced from their respective official implementations.
