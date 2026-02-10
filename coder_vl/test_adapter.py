"""
Test script to verify projection adapter shape transformation.

Runs locally without GPU to verify adapter implementation before SLURM job.
"""

import sys
sys.path.append('.')

from projector import test_projector


if __name__ == "__main__":
    print("="*60)
    print("Testing Projection Adapter")
    print("="*60)
    print()

    adapter = test_projector()

    print()
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"✅ Adapter shape transformation verified")
    print(f"✅ Input: [batch, 1120, 1280] (visual features)")
    print(f"✅ Output: [batch, 1120, 2048] (coder embeddings)")
    print(f"✅ Parameters: {adapter.num_parameters():,}")
    print()
    print("Adapter is ready for Phase 2a training!")
