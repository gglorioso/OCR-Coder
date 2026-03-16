"""
dataset_1_9.py — Phase 1.9: KeywordPatchDataset

Each sample:
  vision  [1024, 1152] float32  — raw 32×32 SigLIP patch features
  labels  [256,  16]   float32  — keyword multi-hot labels aggregated to 16×16
                                   via max-pool over 2×2 input patches
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class KeywordPatchDataset(Dataset):
    """
    Args:
        ground_truth_jsonl : Path to MVV/Phase_1_9/data/ground_truth.jsonl
        feat_dir           : Dir of [1024, 1152] fp16 .pt feature files
        labels_dir         : Dir of [1024, 16] uint8 .pt label files
    """

    def __init__(
        self,
        ground_truth_jsonl: Path | str,
        feat_dir:           Path | str,
        labels_dir:         Path | str,
    ) -> None:
        super().__init__()
        self.feat_dir   = Path(feat_dir)
        self.labels_dir = Path(labels_dir)

        self.entries: list[dict] = []
        with open(ground_truth_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Optional[dict]:
        entry = self.entries[idx]
        stem  = entry["stem"]

        feat_path  = self.feat_dir   / f"{stem}.pt"
        label_path = self.labels_dir / f"{stem}.pt"

        if not feat_path.exists() or not label_path.exists():
            return None

        # Vision features: [1024, 1152] fp16 → float32
        vision = torch.load(feat_path, map_location="cpu", weights_only=True)
        if vision.dtype != torch.float32:
            vision = vision.float()
        if vision.shape != (1024, 1152):
            return None

        # Labels: [1024, 16] uint8 → float32
        labels = torch.load(label_path, map_location="cpu", weights_only=True).float()
        if labels.shape != (1024, 16):
            return None

        # Aggregate 1024 → 256 via spatial max-pool (2×2 blocks)
        # [1024, 16] → [32, 32, 16] → [16, 2, 16, 2, 16] → max → [16, 16, 16] → [256, 16]
        labels_2d  = labels.reshape(32, 32, 16)
        labels_16  = labels_2d.reshape(16, 2, 16, 2, 16).amax(dim=(1, 3))  # [16, 16, 16]
        labels_256 = labels_16.reshape(256, 16)

        return {
            "vision": vision,        # [1024, 1152]
            "labels": labels_256,    # [256, 16]
        }

    @staticmethod
    def collate_fn(batch: list) -> Optional[dict]:
        valid = [item for item in batch if item is not None]
        if not valid:
            return None
        return {
            "vision": torch.stack([item["vision"] for item in valid]),   # [B, 1024, 1152]
            "labels": torch.stack([item["labels"] for item in valid]),   # [B, 256, 16]
        }
