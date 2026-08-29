"""Joint Cross-Attention (JCA), paper Eq. (22)-(27)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_tokens(feat: torch.Tensor) -> torch.Tensor:
    return feat.flatten(2).transpose(1, 2).contiguous()


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.proj = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        q = self.to_q(x).view(b, n, h, d // h).transpose(1, 2)
        k = self.to_k(context).view(b, -1, h, d // h).transpose(1, 2)
        v = self.to_v(context).view(b, -1, h, d // h).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class JointCrossAttention(nn.Module):
    """Independent mask/image cross-attention fused by a 2-layer MLP."""

    def __init__(self, query_dim: int, context_dim: int, heads: int = 4):
        super().__init__()
        self.attn_mask = CrossAttention(query_dim, context_dim, heads=heads)
        self.attn_image = CrossAttention(query_dim, context_dim, heads=heads)
        hidden = 4 * query_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, query_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        z_m = self.attn_mask(x, mask_tokens)
        z_i = self.attn_image(x, image_tokens)
        fused = self.mlp(torch.cat([z_m, z_i], dim=-1))
        return x + fused
