"""
Contrastive Adapter for Phase 1.8.

Architecture:
  1. 2D RoPE injection — encodes the (row, col) grid position of each of the
     64 tokens into the 1152-D feature space.
  2. MLP adapter  — Linear(1152,1152) → GELU → Linear(1152,1152).
  3. Dot-product similarity between adapted vision tokens and the text embedding.

Output: raw logits [B, 64] — BCEWithLogitsLoss is applied in the training loop.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D RoPE helpers
# ---------------------------------------------------------------------------

def _sinusoidal_freqs(seq_len: int, half_dim: int, device: torch.device) -> torch.Tensor:
    """
    Return precomputed (cos, sin) tensors for standard 1-D RoPE.

    Returns shape [seq_len, half_dim] each.
    """
    # θ_i = 1 / 10000^(2i / dim),  dim = 2 * half_dim
    i = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta = 1.0 / (10000.0 ** (i / half_dim))          # [half_dim]
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, theta)               # [seq_len, half_dim]
    return angles.cos(), angles.sin()                    # each [seq_len, half_dim]


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation to x.

    Args:
        x:   [..., seq_len, 2*half_dim]
        cos: [seq_len, half_dim]
        sin: [seq_len, half_dim]

    Returns tensor of same shape as x.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]               # each [..., seq_len, half_dim]
    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
    )


def apply_2d_rope(x: torch.Tensor) -> torch.Tensor:
    """
    Inject 2D positional information into x via RoPE.

    The 64 tokens are laid out in an 8×8 row-major grid:
      token i  →  row = i // 8,  col = i % 8

    Dimension split:
      - dims [0:576]   carry Y-axis (row) RoPE  (576 = 1152/2)
      - dims [576:1152] carry X-axis (col) RoPE

    Args:
        x: [B, 64, 1152]

    Returns:
        [B, 64, 1152]  with RoPE applied in-place (output is a new tensor)
    """
    B, T, D = x.shape
    assert T == 64 and D == 1152, f"Expected [B,64,1152], got {x.shape}"

    half_D = D // 2           # 576 dims per axis
    half_dim = half_D // 2    # 288 pairs per axis

    device = x.device

    # Grid positions for each of the 64 tokens
    rows = torch.arange(64, device=device) // 8  # [64]
    cols = torch.arange(64, device=device) % 8   # [64]

    # Precompute cos/sin tables for positions 0-7
    cos_table, sin_table = _sinusoidal_freqs(8, half_dim, device)
    # cos_table, sin_table: [8, 288]

    # Gather the per-token cos/sin values
    cos_row = cos_table[rows]  # [64, 288]
    sin_row = sin_table[rows]
    cos_col = cos_table[cols]  # [64, 288]
    sin_col = sin_table[cols]

    # Split x into two halves
    x_row = x[:, :, :half_D]    # [B, 64, 576]
    x_col = x[:, :, half_D:]    # [B, 64, 576]

    # Apply RoPE — broadcast batch dimension
    x_row_rot = _rope_rotate(x_row, cos_row, sin_row)  # [B, 64, 576]
    x_col_rot = _rope_rotate(x_col, cos_col, sin_col)  # [B, 64, 576]

    return torch.cat([x_row_rot, x_col_rot], dim=-1)   # [B, 64, 1152]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ContrastiveAdapter(nn.Module):
    """
    Lightweight adapter that learns to map frozen SigLIP vision tokens into a
    space where the text embedding of "def name(" has high dot-product
    similarity with the tokens that visually cover that function.
    """

    def __init__(self, hidden_dim: int = 1152) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        vision: torch.Tensor,   # [B, 64, 1152]
        text: torch.Tensor,     # [B, 1152]
    ) -> torch.Tensor:
        """
        Returns raw logits [B, 64].  BCEWithLogitsLoss is applied externally.
        """
        # 1. Inject 2D positional information
        vision = apply_2d_rope(vision)          # [B, 64, 1152]

        # 2. MLP adaptation
        vision = self.mlp(vision)               # [B, 64, 1152]

        # 3. L2-normalise both modalities
        vision_norm = F.normalize(vision, dim=-1)   # [B, 64, 1152]
        text_norm = F.normalize(text, dim=-1)        # [B, 1152]

        # 4. Per-token dot product with the text embedding
        sim = torch.einsum("bte,be->bt", vision_norm, text_norm)  # [B, 64]
        return sim
