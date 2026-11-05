"""DiT building blocks (Peebles & Xie, 2023), shared by the classification ViT (vit.py) and the generation DiT.

Pure torch: no transformers dependency.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    """The repo's AdaLN: SiLU(Linear(c)). DiT's own form is Linear(SiLU(c))."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear(x))


# --- DiT-S/2 for pixel-space generation ---


@dataclass(frozen=True)
class DiTConfig:
    """Defaults are DiT-S/2 (Peebles & Xie 2023, Table 1) on 32×32×3 CIFAR-10 pixels."""

    depth: int = 12
    hidden_size: int = 384
    num_heads: int = 6
    patch_size: int = 2
    image_size: int = 32
    in_channels: int = 3
    num_classes: int = 10
    mlp_ratio: float = 4.0


def get_2d_sincos_pos_embed(dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2-D sin-cos position embedding (MAE/DiT): half the channels encode the
    row, half the column. Returns (grid_size², dim) in row-major patch order.
    float32 throughout (MPS has no float64); positions ≤ grid_size keep it exact to ~1e-6."""

    def embed_1d(pos: torch.Tensor, d: int) -> torch.Tensor:
        omega = 1.0 / 10000 ** (torch.arange(d // 2) / (d / 2))
        out = pos[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    coords = torch.arange(grid_size, dtype=torch.float32)
    rows, cols = torch.meshgrid(coords, coords, indexing="ij")
    return torch.cat(
        [embed_1d(rows.flatten(), dim // 2), embed_1d(cols.flatten(), dim // 2)], dim=1
    )


def unpatchify(x: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    """(N, T, p·p·C) tokens → (N, C, H, W) image."""
    n, t, _ = x.shape
    h = w = int(t**0.5)
    x = x.reshape(n, h, w, patch_size, patch_size, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(n, channels, h * patch_size, w * patch_size)


class DiTLayer(nn.Module):
    """Pre-LN transformer layer with AdaLN-Zero (shift, scale, gate per sub-layer)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = AdaLN(hidden_size, 6 * hidden_size)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape
        q, k, v = self.qkv(x).view(n, t, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(n, t, d))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = AdaLN(hidden_size, 2 * hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """DiT with a stem and head shared by all blocks; `forward_block` runs only the
    given layers (one DiffusionBlock), like `vit.py`'s `forward_block`."""

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        d, p = config.hidden_size, config.patch_size
        self.x_embedder = nn.Conv2d(config.in_channels, d, kernel_size=p, stride=p)
        grid_size = config.image_size // p
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(d, grid_size)[None])
        self.t_embedder = TimestepEmbedder(d)
        self.y_embedder = nn.Embedding(config.num_classes, d)
        self.layers = nn.ModuleList(
            DiTLayer(d, config.num_heads, config.mlp_ratio) for _ in range(config.depth)
        )
        self.final_layer = FinalLayer(d, p, config.in_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """DiT's initialization: Xavier linears, N(0, 0.02) embedders, AdaLN-Zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        w, b = self.x_embedder.weight, self.x_embedder.bias
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if b is not None:
            nn.init.zeros_(b)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        for m in self.t_embedder.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for module in [*self.layers, self.final_layer]:
            nn.init.zeros_(module.adaLN_modulation.linear.weight)
            nn.init.zeros_(module.adaLN_modulation.linear.bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward_block(
        self,
        x: torch.Tensor,
        c_noise: torch.Tensor,
        y: torch.Tensor,
        keep: torch.Tensor,
        layer_indices: list[int],
    ) -> torch.Tensor:
        """F(x; c_noise, y) through the stem, the given layers and the head.
        `keep` (N,) is 1 for conditional, 0 for the null class (zeroed embedding)."""
        h = self.x_embedder(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = F.silu(self.t_embedder(c_noise) + keep[:, None] * self.y_embedder(y))
        for i in layer_indices:
            h = self.layers[i](h, c)
        h = self.final_layer(h, c)
        return unpatchify(h, self.config.patch_size, self.config.in_channels)
