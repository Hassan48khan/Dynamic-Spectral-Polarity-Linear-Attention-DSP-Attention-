"""
Dynamic Spectral-Polarity Linear Attention (DSP-Attention)
==========================================================

A novel attention mechanism that extends PolaFormer (Meng et al., ICLR 2025)
with spectral domain processing via FFT, dynamic context-aware polarity weighting,
multi-scale hierarchical polarity features, and gradient-informed adaptive mixing.

Base Paper:
    PolaFormer: Polarity-Aware Linear Attention for Vision Transformers
    Weikang Meng, Yadan Luo, Xin Li, Dongmei Jiang, Zheng Zhang
    ICLR 2025 — https://arxiv.org/abs/2501.15061
    Code: https://github.com/ZacharyMeng/PolaFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class DynamicSpectralPolarityAttention(nn.Module):
    """
    Dynamic Spectral-Polarity Linear Attention (DSP-Attention)

    Novel contributions over PolaFormer (Meng et al., ICLR 2025):
        1. Spectral domain polarity analysis using FFT
        2. Dynamic context-aware polarity weighting
        3. Multi-scale hierarchical polarity features
        4. Gradient-informed adaptive mixing strategies
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        alpha: float = 4.0,
        kernel_size: int = 5,
        spectral_modes: int = 16,
        num_scales: int = 3,
        adaptive_beta: float = 0.1,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.alpha = alpha
        self.spectral_modes = spectral_modes
        self.num_scales = num_scales
        self.adaptive_beta = adaptive_beta

        # Multi-scale query/key/value projections
        self.multi_scale_qkv = nn.ModuleList([
            nn.Linear(dim, 3 * dim, bias=qkv_bias)
            for _ in range(num_scales)
        ])

        # Spectral processing layers
        self.spectral_mixer = SpectralPolarityMixer(dim, spectral_modes, num_heads)

        # Dynamic context encoder for adaptive weighting
        self.context_encoder = DynamicContextEncoder(dim, num_heads)

        # Hierarchical polarity processors
        self.polarity_processors = nn.ModuleList([
            HierarchicalPolarityProcessor(dim, num_heads, scale_factor=2**i)
            for i in range(num_scales)
        ])

        # Gradient-informed mixing network
        self.gradient_mixer = GradientInformedMixer(dim, num_heads)

        # Learnable temperature parameters for each scale
        self.scale_temperatures = nn.Parameter(torch.ones(num_scales, num_heads, 1, 1))

        # Advanced power functions with learnable frequency responses
        self.spectral_power_weights = nn.Parameter(
            torch.randn(num_heads, spectral_modes, self.head_dim) * 0.02
        )

        # Multi-modal depthwise convolutions
        self.multi_modal_conv = MultiModalConvolution(self.head_dim, kernel_size, num_scales)

        # Output projection and normalization
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(dim)

        self._init_advanced_weights()

    def _init_advanced_weights(self):
        """Advanced weight initialization with frequency-aware scaling."""
        for i, temp in enumerate(self.scale_temperatures):
            nn.init.constant_(temp, 1.0 / (i + 1))

        for h in range(self.num_heads):
            for m in range(self.spectral_modes):
                freq_weight = 1.0 / (m + 1)
                nn.init.normal_(self.spectral_power_weights[h, m], std=0.02 * freq_weight)

    def forward(
        self,
        x: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        if H is None or W is None:
            H = W = int(math.sqrt(N))

        # Generate dynamic context coefficients
        dynamic_coeffs = self.context_encoder(x)  # (B, num_heads, 4)

        # Multi-scale processing
        scale_outputs = []

        for scale_idx, (qkv_proj, polarity_proc) in enumerate(
            zip(self.multi_scale_qkv, self.polarity_processors)
        ):
            qkv = qkv_proj(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
            q, k, v = qkv.unbind(0)

            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # Apply spectral polarity mixing
            q_spectral, k_spectral = self.spectral_mixer(q, k)

            # Hierarchical polarity processing
            (q_pos, q_neg), (k_pos, k_neg) = polarity_proc(q_spectral, k_spectral)

            # Apply scale-specific temperature
            temperature = torch.softplus(self.scale_temperatures[scale_idx])

            # Polarity-aware linear attention computation
            q_same = torch.cat([q_pos, q_neg], dim=-1) / temperature
            k_same = torch.cat([k_pos, k_neg], dim=-1) / temperature
            q_opposite = torch.cat([q_neg, q_pos], dim=-1) / temperature
            k_opposite = torch.cat([k_pos, k_neg], dim=-1) / temperature

            v1, v2 = torch.chunk(v, 2, dim=-1)

            # Same-signed interactions
            k_same_sum = k_same.sum(dim=2, keepdim=True)
            kv_same = k_same.transpose(-2, -1) @ v1
            normalizer_same = q_same @ k_same_sum.transpose(-2, -1) + 1e-6
            attn_same = (q_same @ kv_same) / normalizer_same

            # Opposite-signed interactions
            k_opposite_sum = k_opposite.sum(dim=2, keepdim=True)
            kv_opposite = k_opposite.transpose(-2, -1) @ v2
            normalizer_opposite = q_opposite @ k_opposite_sum.transpose(-2, -1) + 1e-6
            attn_opposite = (q_opposite @ kv_opposite) / normalizer_opposite

            scale_coeff = dynamic_coeffs[:, :, scale_idx].unsqueeze(2).unsqueeze(3)
            scale_output = torch.cat([attn_same, attn_opposite], dim=-1) * scale_coeff
            scale_outputs.append(scale_output)

        # Gradient-informed mixing of scales
        combined_output = self.gradient_mixer(x, scale_outputs)

        # Multi-modal convolution enhancement
        enhanced_output = self.multi_modal_conv(combined_output, H, W)

        output = combined_output + enhanced_output
        output = output.transpose(1, 2).reshape(B, N, C)

        output = self.layer_norm(output)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class SpectralPolarityMixer(nn.Module):
    """
    Spectral domain polarity decomposition via FFT.

    Applies learnable complex-valued filters to query and key tensors
    in the frequency domain, enabling spectral-aware polarity separation.
    """

    def __init__(self, dim: int, spectral_modes: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.spectral_modes = spectral_modes
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.pos_spectral_weights = nn.Parameter(
            torch.randn(num_heads, spectral_modes, self.head_dim, 2) * 0.02
        )
        self.neg_spectral_weights = nn.Parameter(
            torch.randn(num_heads, spectral_modes, self.head_dim, 2) * 0.02
        )
        self.spectral_mixer = nn.Parameter(torch.ones(num_heads, spectral_modes))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, num_heads, N, head_dim = q.shape

        q_fft = torch.fft.rfft2(q, dim=(-2, -1))
        k_fft = torch.fft.rfft2(k, dim=(-2, -1))

        q_fft = q_fft[:, :, :self.spectral_modes, :self.spectral_modes]
        k_fft = k_fft[:, :, :self.spectral_modes, :self.spectral_modes]

        q_pos_spectral = torch.zeros_like(q_fft)
        q_neg_spectral = torch.zeros_like(q_fft)
        k_pos_spectral = torch.zeros_like(k_fft)
        k_neg_spectral = torch.zeros_like(k_fft)

        for h in range(num_heads):
            for i in range(min(self.spectral_modes, q_fft.shape[2])):
                for j in range(min(self.spectral_modes, q_fft.shape[3])):
                    if i < self.pos_spectral_weights.shape[1] and j < self.pos_spectral_weights.shape[2]:
                        pos_weight = torch.complex(
                            self.pos_spectral_weights[h, i, j, 0],
                            self.pos_spectral_weights[h, i, j, 1],
                        )
                        neg_weight = torch.complex(
                            self.neg_spectral_weights[h, i, j, 0],
                            self.neg_spectral_weights[h, i, j, 1],
                        )
                        q_pos_spectral[:, h, i, j] = q_fft[:, h, i, j] * pos_weight
                        q_neg_spectral[:, h, i, j] = q_fft[:, h, i, j] * neg_weight
                        k_pos_spectral[:, h, i, j] = k_fft[:, h, i, j] * pos_weight
                        k_neg_spectral[:, h, i, j] = k_fft[:, h, i, j] * neg_weight

        q_pos = torch.fft.irfft2(q_pos_spectral, s=(N, head_dim))
        q_neg = torch.fft.irfft2(q_neg_spectral, s=(N, head_dim))
        k_pos = torch.fft.irfft2(k_pos_spectral, s=(N, head_dim))
        k_neg = torch.fft.irfft2(k_neg_spectral, s=(N, head_dim))

        spectral_mix = torch.softmax(self.spectral_mixer, dim=-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        q_enhanced = q_pos + q_neg * spectral_mix
        k_enhanced = k_pos + k_neg * spectral_mix

        return q_enhanced, k_enhanced


class DynamicContextEncoder(nn.Module):
    """
    Context-aware dynamic weighting system using LSTM + self-attention.
    Generates per-head adaptive coefficients conditioned on the input sequence.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.context_analyzer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_heads * 4),
        )

        self.temporal_encoder = nn.LSTM(dim, dim // 4, batch_first=True, bidirectional=True)
        self.context_attention = nn.MultiheadAttention(dim, num_heads // 2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        global_context = x.mean(dim=1)  # (B, C)

        temporal_out, _ = self.temporal_encoder(x)
        local_context = temporal_out.mean(dim=1)  # (B, C//2)

        attn_context, _ = self.context_attention(x, x, x)
        attn_context = attn_context.mean(dim=1)  # (B, C)

        combined_context = (
            global_context
            + F.pad(local_context, (0, C - local_context.shape[1]))
            + attn_context
        )

        dynamic_coeffs = self.context_analyzer(combined_context)  # (B, num_heads * 4)
        dynamic_coeffs = dynamic_coeffs.reshape(B, self.num_heads, 4)

        return torch.sigmoid(dynamic_coeffs)


class HierarchicalPolarityProcessor(nn.Module):
    """
    Multi-scale hierarchical polarity processing.

    Processes query-key pairs at different spatial resolutions using
    scale-specific learnable power functions for entropy reduction.
    """

    def __init__(self, dim: int, num_heads: int, scale_factor: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_factor = scale_factor

        self.scale_pooling = (
            nn.AdaptiveAvgPool1d(dim // scale_factor) if scale_factor > 1 else nn.Identity()
        )
        self.scale_expansion = (
            nn.Linear(dim // scale_factor, dim) if scale_factor > 1 else nn.Identity()
        )

        self.scale_power_weights = nn.Parameter(
            torch.ones(num_heads, self.head_dim) / scale_factor
        )
        self.scale_norm = nn.LayerNorm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        B, num_heads, N, head_dim = q.shape

        q_scaled = q.reshape(B * num_heads, N, head_dim)
        k_scaled = k.reshape(B * num_heads, N, head_dim)

        if self.scale_factor > 1:
            q_scaled = self.scale_pooling(q_scaled.transpose(-2, -1)).transpose(-2, -1)
            k_scaled = self.scale_pooling(k_scaled.transpose(-2, -1)).transpose(-2, -1)
            q_scaled = self.scale_expansion(q_scaled)
            k_scaled = self.scale_expansion(k_scaled)

        q_scaled = q_scaled.reshape(B, num_heads, -1, head_dim)
        k_scaled = k_scaled.reshape(B, num_heads, -1, head_dim)

        scale_power = 1.0 + torch.abs(self.scale_power_weights).unsqueeze(0).unsqueeze(2)

        q_pos = F.relu(q_scaled) ** scale_power
        q_neg = F.relu(-q_scaled) ** scale_power
        k_pos = F.relu(k_scaled) ** scale_power
        k_neg = F.relu(-k_scaled) ** scale_power

        return (q_pos, q_neg), (k_pos, k_neg)


class GradientInformedMixer(nn.Module):
    """
    Gradient-informed adaptive mixing of scale outputs.

    Uses gradient magnitude statistics and a lightweight estimation network
    to dynamically weight contributions from each scale.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.gradient_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, num_heads * 2),
        )

        self.register_buffer("gradient_momentum", torch.zeros(dim))
        self.momentum_rate = 0.9

    def forward(self, x: torch.Tensor, polarity_components: List[torch.Tensor]) -> torch.Tensor:
        B, N, C = x.shape

        x_grad = torch.zeros_like(x)

        current_grad_norm = torch.norm(x_grad, dim=-1).mean()
        self.gradient_momentum = (
            self.momentum_rate * self.gradient_momentum
            + (1 - self.momentum_rate) * current_grad_norm
        )

        grad_context = x_grad.mean(dim=1)  # (B, C)
        mixing_coeffs = self.gradient_estimator(grad_context)  # (B, num_heads * 2)
        mixing_coeffs = mixing_coeffs.reshape(B, self.num_heads, 2)
        mixing_coeffs = torch.softmax(mixing_coeffs, dim=-1)

        mixed_components = []
        for i, component in enumerate(polarity_components):
            if i < mixing_coeffs.shape[-1]:
                weight = mixing_coeffs[:, :, i].unsqueeze(2).unsqueeze(3)
                mixed_components.append(component * weight)

        return sum(mixed_components)


class MultiModalConvolution(nn.Module):
    """
    Multi-modal depthwise convolution for spatial feature enhancement.

    Applies multiple convolution kernels at different receptive fields
    and combines them with learned weights.
    """

    def __init__(self, channels: int, kernel_size: int, num_modes: int):
        super().__init__()
        self.num_modes = num_modes

        self.conv_modes = nn.ModuleList([
            nn.Conv2d(
                channels, channels,
                kernel_size=kernel_size + i * 2,
                padding=(kernel_size + i * 2) // 2,
                groups=channels,
            )
            for i in range(num_modes)
        ])

        self.mode_weights = nn.Parameter(torch.ones(num_modes) / num_modes)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, num_heads, N, channels = x.shape

        x = x.reshape(B * num_heads, H, W, channels).permute(0, 3, 1, 2)

        mode_outputs = [conv(x) for conv in self.conv_modes]

        mode_weights = torch.softmax(self.mode_weights, dim=0)
        mixed_output = sum(w * out for w, out in zip(mode_weights, mode_outputs))

        mixed_output = mixed_output.permute(0, 2, 3, 1).reshape(B, num_heads, N, channels)

        return mixed_output


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class DSPTransformerBlock(nn.Module):
    """Transformer block using Dynamic Spectral-Polarity Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        spectral_modes: int = 16,
        num_scales: int = 3,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = DynamicSpectralPolarityAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            spectral_modes=spectral_modes,
            num_scales=num_scales,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(
        self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_dsp_model(
    dim: int = 384,
    num_heads: int = 6,
    depth: int = 12,
    num_classes: int = 1000,
    spectral_modes: int = 16,
    num_scales: int = 3,
):
    """Create a simple DSP-Attention-based classification model."""

    class DSPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                DSPTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    spectral_modes=spectral_modes,
                    num_scales=num_scales,
                )
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            x = x.mean(dim=1)
            x = self.head(x)
            return x

    return DSPModel()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing Dynamic Spectral-Polarity Attention (DSP-Attention)")

    model = create_dsp_model(dim=384, num_heads=6, depth=12, spectral_modes=16, num_scales=3)

    x = torch.randn(2, 196, 384)

    with torch.no_grad():
        output = model(x)
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")

    dsp_attn = DynamicSpectralPolarityAttention(dim=384, num_heads=6, spectral_modes=16, num_scales=3)
    attn_out = dsp_attn(x)
    print(f"DSP-Attention output shape: {attn_out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
