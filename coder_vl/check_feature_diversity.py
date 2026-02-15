"""
Check if pre-computed visual features are actually diverse across different images.
If all features are similar, the model has no signal to learn from.
"""

import torch
from pathlib import Path
import numpy as np


def main():
    feat_dir = Path("./precomputed_features")
    feat_files = sorted(feat_dir.glob("*.pt"))[:20]  # Check first 20

    print("=" * 80)
    print("CHECKING VISUAL FEATURE DIVERSITY")
    print("=" * 80)
    print(f"\nLoading {len(feat_files)} feature files...\n")

    features_list = []
    names = []

    for f in feat_files:
        feat = torch.load(f, map_location="cpu")
        features_list.append(feat)
        names.append(f.stem)
        print(f"{f.stem[:40]:40s}  shape={str(feat.shape):15s}  "
              f"mean={feat.mean():.4f}  std={feat.std():.4f}")

    # Compute pairwise cosine similarities
    print(f"\n{'=' * 80}")
    print("PAIRWISE COSINE SIMILARITIES (first 10 pairs)")
    print("=" * 80)
    print("If all images have similarity ~0.9+, features are too similar!\n")

    for i in range(min(10, len(features_list))):
        for j in range(i + 1, min(10, len(features_list))):
            feat_i = features_list[i].flatten()
            feat_j = features_list[j].flatten()

            # Pad to same length if needed
            if len(feat_i) != len(feat_j):
                min_len = min(len(feat_i), len(feat_j))
                feat_i = feat_i[:min_len]
                feat_j = feat_j[:min_len]

            # Cosine similarity
            cos_sim = torch.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0)).item()

            name_i = names[i][:20]
            name_j = names[j][:20]
            print(f"  {name_i:20s} <-> {name_j:20s}  sim={cos_sim:.4f}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print("=" * 80)
    print("If similarities are all > 0.9: Features are nearly identical (BAD)")
    print("If similarities are 0.3-0.7: Features are diverse (GOOD)")
    print("=" * 80)


if __name__ == "__main__":
    main()
