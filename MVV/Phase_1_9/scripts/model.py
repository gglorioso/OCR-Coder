"""
model.py — Phase 1.9: ConvRoPEProjector + LinearProbe

Architecture:
  ConvRoPEProjector  [B, 1024, 1152] → [B, 256, 2048]
    1. reshape  → [B, 1152, 32, 32]
    2. Conv2d(1152, 1152, k=2, stride=2)  → [B, 1152, 16, 16]
    3. flatten + transpose  → [B, 256, 1152]
    4. 2D RoPE (16×16 grid)  → [B, 256, 1152]
    5. MLP: Linear(1152,2048) → GELU → Linear(2048,2048)  → [B, 256, 2048]

  LinearProbe  [B, 256, 2048] → [B, 256, VOCAB_SIZE]

  ConvRoPEKeywordDetector  = ConvRoPEProjector + LinearProbe
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KEYWORDS = [
    'def', 'class', 'import', 'return',
    'if',  'for',   'while',  'else',
    'elif','try',   'except', 'with',
    'pass','yield', 'lambda', 'raise',
]
VOCAB_SIZE = len(KEYWORDS)   # 16


# ---------------------------------------------------------------------------
# 2D RoPE helpers (16×16 grid, 1152-D tokens)
# ---------------------------------------------------------------------------

def _sinusoidal_freqs(seq_len: int, half_dim: int, device: torch.device):
    """Return (cos, sin) tables each [seq_len, half_dim]."""
    i      = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta  = 1.0 / (10000.0 ** (i / half_dim))
    pos    = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, theta)          # [seq_len, half_dim]
    return angles.cos(), angles.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation. x: [..., seq_len, 2*half_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_2d_rope_16x16(x: torch.Tensor) -> torch.Tensor:
    """
    Inject 2D positional information into x via RoPE.

    The 256 tokens are laid out in a 16×16 row-major grid:
      token i  →  row = i // 16,  col = i % 16

    Dimension split (D=1152):
      dims [0:576]    carry Y-axis (row) RoPE
      dims [576:1152] carry X-axis (col) RoPE

    Args:
        x: [B, 256, 1152]
    Returns:
        [B, 256, 1152]
    """
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"

    half_D   = D // 2        # 576
    half_dim = half_D // 2   # 288 sin/cos pairs per axis
    device   = x.device

    rows = torch.arange(256, device=device) // 16   # [256]
    cols = torch.arange(256, device=device) % 16    # [256]

    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
    # [16, 288]

    cos_row = cos_table[rows]   # [256, 288]
    sin_row = sin_table[rows]
    cos_col = cos_table[cols]   # [256, 288]
    sin_col = sin_table[cols]

    x_row = x[:, :, :half_D]    # [B, 256, 576]
    x_col = x[:, :, half_D:]    # [B, 256, 576]

    x_row_rot = _rope_rotate(x_row, cos_row, sin_row)
    x_col_rot = _rope_rotate(x_col, cos_col, sin_col)

    return torch.cat([x_row_rot, x_col_rot], dim=-1)   # [B, 256, 1152]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ConvRoPEProjector(nn.Module):
    """
    Compresses [B, 1024, 1152] raw SigLIP tokens to [B, 256, 2048] via:
      strided conv (32→16)  →  2D RoPE  →  MLP (1152→2048)
    """

    def __init__(self, feat_dim: int = 1152, proj_dim: int = 2048) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.grid_in  = 32
        self.grid_out = 16

        self.conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=2, stride=2)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1024, 1152]  →  [B, 256, 2048]"""
        B, N, C = x.shape
        assert N == self.grid_in ** 2 and C == self.feat_dim, \
            f"Expected [B,{self.grid_in**2},{self.feat_dim}], got {x.shape}"

        # 1. Reshape to spatial grid
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        # [B, 1152, 32, 32]

        # 2. Stride-2 conv: 32×32 → 16×16
        x = self.conv(x)
        # [B, 1152, 16, 16]

        # 3. Flatten spatial dims back to sequence
        x = x.flatten(2).transpose(1, 2)
        # [B, 256, 1152]

        # 4. Inject 2D positional information via RoPE
        x = apply_2d_rope_16x16(x)
        # [B, 256, 1152]

        # 5. Project to LLM embedding dimension
        x = self.mlp(x)
        # [B, 256, 2048]

        return x


class LinearProbe(nn.Module):
    """Single linear layer mapping each token to keyword logits."""

    def __init__(self, proj_dim: int = 2048, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.probe = nn.Linear(proj_dim, vocab_size)
        nn.init.xavier_uniform_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 256, 2048]  →  [B, 256, VOCAB_SIZE]"""
        return self.probe(x)


class ConvRoPEKeywordDetector(nn.Module):
    """Full model: ConvRoPEProjector + LinearProbe."""

    def __init__(self, feat_dim: int = 1152, proj_dim: int = 2048,
                 vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.projector = ConvRoPEProjector(feat_dim=feat_dim, proj_dim=proj_dim)
        self.probe     = LinearProbe(proj_dim=proj_dim, vocab_size=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1024, 1152]  →  logits [B, 256, VOCAB_SIZE]"""
        return self.probe(self.projector(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
