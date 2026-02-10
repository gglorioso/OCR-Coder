#!/usr/bin/env python3
"""
inspect_coder_embeddings.py — Inspect DeepSeek-Coder-V2-Lite embedding dimension

We already know from Phase 1 that the vision encoder outputs 1280-dimensional embeddings.
This script focuses only on determining the coder model's embedding dimension.

Expected output:
  - Coder-V2-Lite: ?-dimensional embeddings (to be determined)

Usage:
    python inspect_coder_embeddings.py
    # or via SLURM:
    sbatch inspect_coder_embeddings.sh
"""

import sys
import torch

print("=" * 70)
print("  Inspect DeepSeek-Coder-V2-Lite Embedding Dimension")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    free_mem = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU Memory Available: {free_mem:.1f} GB")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Inspect Code Reasoning Model (DeepSeek-Coder-V2-Lite)
# ═══════════════════════════════════════════════════════════════════════════

print("Loading DeepSeek-Coder-V2-Lite-Instruct...")
print()

try:
    from transformers import AutoConfig

    coder_model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    print(f"  Model: {coder_model_name}")
    print("  (Loading config only - no model weights needed)")
    print()

    # Load config only (no weights - much faster and uses minimal memory)
    print("  [1/1] Loading model config...")
    config = AutoConfig.from_pretrained(coder_model_name, trust_remote_code=True)
    print(f"  ✓ Config loaded")
    print()

    # Inspect the configuration
    print("  Model configuration:")
    print(f"    - Config type: {type(config).__name__}")

    # Extract embedding dimension
    coder_embed_dim = None
    if hasattr(config, 'hidden_size'):
        coder_embed_dim = config.hidden_size
        print(f"    - hidden_size: {config.hidden_size}")
    if hasattr(config, 'd_model'):
        print(f"    - d_model: {config.d_model}")
    if hasattr(config, 'n_embd'):
        print(f"    - n_embd: {config.n_embd}")
    if hasattr(config, 'vocab_size'):
        print(f"    - vocab_size: {config.vocab_size}")
    if hasattr(config, 'num_hidden_layers'):
        print(f"    - num_hidden_layers: {config.num_hidden_layers}")
    if hasattr(config, 'num_attention_heads'):
        print(f"    - num_attention_heads: {config.num_attention_heads}")
    print()

    if coder_embed_dim:
        print("  " + "─" * 66)
        print(f"  Embedding Dimension (from config): {coder_embed_dim}")
        print("  " + "─" * 66)
        print()
        print("  ℹ️  Note: This is from the config file, not loaded weights")
        print("     The config dimension is reliable for designing the adapter")
        print()
    else:
        print("  ⚠️  Could not find embedding dimension in config")
        print()

    coder_success = coder_embed_dim is not None

except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print()
    coder_embed_dim = None
    coder_success = False


# ═══════════════════════════════════════════════════════════════════════════
# Summary & Projection Adapter Design
# ═══════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  📊 SUMMARY")
print("=" * 70)
print()

# Vision encoder dimension from Phase 1
vision_embed_dim = 1280
print(f"  Vision Encoder Output:  {vision_embed_dim:>6d} dimensions  (from Phase 1)")

if coder_success and coder_embed_dim:
    print(f"  Code Model Input:       {coder_embed_dim:>6d} dimensions  ✅")
else:
    print(f"  Code Model Input:       {'FAILED':>6s}")

print()

if coder_success and coder_embed_dim:
    print("  " + "─" * 66)
    print("  PROJECTION ADAPTER DESIGN:")
    print("  " + "─" * 66)
    print()
    print(f"  The adapter must map: {vision_embed_dim}D → {coder_embed_dim}D")
    print()

    # Suggest architecture
    intermediate_dim = max(vision_embed_dim, coder_embed_dim) * 2

    print("  Recommended 2-layer MLP architecture:")
    print()
    print("    ```python")
    print("    import torch.nn as nn")
    print()
    print("    projector = nn.Sequential(")
    print(f"        nn.Linear({vision_embed_dim}, {intermediate_dim}),  # Layer 1")
    print("        nn.GELU(),")
    print(f"        nn.Linear({intermediate_dim}, {coder_embed_dim}),   # Layer 2")
    print("    )")
    print("    ```")
    print()

    # Calculate parameter count
    params_layer1 = vision_embed_dim * intermediate_dim + intermediate_dim
    params_layer2 = intermediate_dim * coder_embed_dim + coder_embed_dim
    total_params = params_layer1 + params_layer2

    print(f"  Parameter breakdown:")
    print(f"    - Layer 1: {params_layer1 / 1e6:.1f}M parameters")
    print(f"    - Layer 2: {params_layer2 / 1e6:.1f}M parameters")
    print(f"    - Total:   {total_params / 1e6:.1f}M parameters")
    print()

    print("  " + "─" * 66)
    print("  ✅ Ready to proceed to Phase 2: Build projection adapter")
    print("  " + "─" * 66)

else:
    print("  ❌ Failed to determine coder embedding dimension")
    print("     Check error messages above for details")

print()
print("=" * 70)
print("Done!")
print()
