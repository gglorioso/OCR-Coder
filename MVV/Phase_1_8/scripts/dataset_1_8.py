"""
Dataset for Phase 1.8 spatial contrastive training.

Each sample pairs:
  - vision: [64, 1152] float32   — 8×8 pooled SigLIP patch features
  - text:   [1152]    float32   — EOS-pooled SigLIP text embedding for "def name("
  - target_mask: [64] float32   — multi-hot mask; 1.0 for tokens in grid_rows
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class SpatialContrastiveDataset(Dataset):
    """
    Args:
        jsonl_path:    Path to ground_truth.jsonl
        feat_dir:      Directory containing per-file .pt vision features
        text_emb_path: Path to the precomputed text_embeddings.pt dict
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        feat_dir: Path | str,
        text_emb_path: Path | str,
    ) -> None:
        super().__init__()
        self.feat_dir = Path(feat_dir)

        # Load all JSONL entries
        self.entries: list[dict] = []
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

        # Load precomputed text embeddings
        self.text_embeddings: dict[str, torch.Tensor] = torch.load(
            text_emb_path, map_location="cpu", weights_only=True
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_target_mask(grid_rows: list[int]) -> torch.Tensor:
        """
        Build a [64] float32 multi-hot mask.

        The 64 tokens cover an 8×8 grid in row-major order:
          token index = row * 8 + col,  row, col ∈ {0, …, 7}

        All 8 column tokens for each row in `grid_rows` are set to 1.0.
        """
        mask = torch.zeros(64, dtype=torch.float32)
        for row in grid_rows:
            start = row * 8
            mask[start : start + 8] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Optional[dict]:
        entry = self.entries[idx]
        file_key = entry["file"]
        name = entry["name"]
        grid_rows = entry["grid_rows"]

        # ---- Vision features ----
        feat_path = self.feat_dir / f"{file_key}.pt"
        if not feat_path.exists():
            return None

        vision = torch.load(feat_path, map_location="cpu", weights_only=True)
        # Stored as fp16; cast to float32 for numerics
        if vision.dtype != torch.float32:
            vision = vision.float()
        # Expect shape [64, 1152]
        if vision.shape != (64, 1152):
            return None

        # ---- Text embedding ----
        query = f"def {name}("
        if query not in self.text_embeddings:
            return None
        text = self.text_embeddings[query].float()  # [1152]

        # ---- Target mask ----
        target_mask = self._build_target_mask(grid_rows)

        return {
            "vision": vision,         # [64, 1152]
            "text": text,             # [1152]
            "target_mask": target_mask,  # [64]
        }

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[Optional[dict]]) -> Optional[dict]:
        """Filter out None items (missing vision files) and stack the rest."""
        valid = [item for item in batch if item is not None]
        if not valid:
            return None
        return {
            "vision": torch.stack([item["vision"] for item in valid]),        # [B, 64, 1152]
            "text": torch.stack([item["text"] for item in valid]),            # [B, 1152]
            "target_mask": torch.stack([item["target_mask"] for item in valid]),  # [B, 64]
        }
