#!/usr/bin/env python3
"""
inspect_embeddings_v2.py — Inspect embedding dimensions for Phase 2 (Fixed)

This script loads both the vision encoder (DeepSeek-VL2 / DeepSeek-OCR-2)
and the code reasoning model (DeepSeek-Coder-V2-Lite) to inspect their
embedding dimensions. This is necessary to design the projection adapter.

FIXED: Matches Phase 1 approach - no device_map, manual GPU placement

Expected outputs:
  - Vision encoder (SigLIP): 1280-dimensional embeddings
  - Coder-V2-Lite: ?-dimensional embeddings (to be determined)

Usage:
    python inspect_embeddings_v2.py
    # or via SLURM:
    sbatch inspect_embeddings_v2.sh
"""

import sys
import torch
from pathlib import Path

print("=" * 70)
print("  Phase 2 Prep: Embedding Dimension Inspector (v2)")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Inspect Vision Encoder (DeepSeek-OCR-2 / DeepSeek-VL2)
# ═══════════════════════════════════════════════════════════════════════════

print("[1/2] Inspecting Vision Encoder (DeepSeek-OCR-2)...")
print()

try:
    from transformers import AutoModel

    vision_model_name = "deepseek-ai/DeepSeek-OCR-2"
    print(f"  Loading {vision_model_name}...")

    # Load model WITHOUT device_map (like Phase 1)
    try:
        vision_model = AutoModel.from_pretrained(
            vision_model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        print(f"  ✓ Model loaded (Flash Attention 2)")
    except Exception:
        vision_model = AutoModel.from_pretrained(
            vision_model_name,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
        )
        print(f"  ✓ Model loaded (eager attention)")

    # Manually move to GPU (like Phase 1)
    if torch.cuda.is_available():
        vision_model = vision_model.eval().cuda().to(torch.bfloat16)
        print(f"  ✓ Moved to GPU")
    else:
        vision_model = vision_model.eval().to(torch.bfloat16)
        print(f"  ✓ Running on CPU")

    print()

    # Inspect the architecture
    print("  Model structure:")
    inner_model = vision_model.model  # DeepseekOCR2Model

    print(f"    - Main model: {type(inner_model).__name__}")
    print(f"    - Vision encoder (sam_model): {type(inner_model.sam_model).__name__}")
    print(f"    - Vision backbone (qwen2_model): {type(inner_model.qwen2_model).__name__}")
    print(f"    - Projector: {type(inner_model.projector).__name__}")
    print()

    # Get output dimensions by running a test inference
    print("  Testing vision encoder output dimensions...")

    # Create a dummy image (1024x1024 RGB)
    from PIL import Image
    from torchvision import transforms

    dummy_image = Image.new('RGB', (1024, 1024), color=(128, 128, 128))

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    image_tensor = image_transform(dummy_image).unsqueeze(0).to(torch.bfloat16)  # [1, 3, 1024, 1024]

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        # Run through vision pipeline
        sam_output = inner_model.sam_model(image_tensor)
        qwen2_output = inner_model.qwen2_model(sam_output)
        final_output = inner_model.projector(qwen2_output)

    print(f"    Input image tensor shape: {image_tensor.shape}")
    print(f"    SAM output shape: {sam_output.shape}")
    print(f"    Qwen2 output shape: {qwen2_output.shape}")
    print(f"    Projector output shape: {final_output.shape}")
    print()

    # Extract embedding dimension
    vision_embed_dim = final_output.shape[-1]
    vision_tokens = final_output.shape[1]

    print("  " + "─" * 66)
    print(f"  Vision Encoder Output Dimension: {vision_embed_dim}")
    print(f"  Number of tokens (base image): {vision_tokens}")
    print("  " + "─" * 66)

    if vision_embed_dim == 1280:
        print("  ✅ Matches expected dimension (1280)")
    else:
        print(f"  ⚠️  Expected 1280, got {vision_embed_dim}")

    print()

    # Clean up GPU memory BEFORE loading coder model
    print("  Cleaning up GPU memory before loading coder model...")
    del vision_model, inner_model, image_tensor, sam_output, qwen2_output, final_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Print available memory
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f"  ✓ GPU memory freed: {free_mem:.1f} GB / {total_mem:.1f} GB available")
    print()

    vision_success = True

except Exception as e:
    print(f"  ❌ Error loading vision model: {e}")
    import traceback
    traceback.print_exc()
    print()
    vision_embed_dim = None
    vision_success = False
    # Still try to clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Inspect Code Reasoning Model (DeepSeek-Coder-V2-Lite)
# ═══════════════════════════════════════════════════════════════════════════

print("[2/2] Inspecting Code Reasoning Model (DeepSeek-Coder-V2-Lite)...")
print()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    coder_model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    print(f"  Loading {coder_model_name}...")
    print("  (This may take a few minutes — it's a 16B parameter model)")
    print()

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(coder_model_name, trust_remote_code=True)
    print(f"  ✓ Tokenizer loaded")
    print(f"    - Vocab size: {tokenizer.vocab_size}")
    print()

    # Load the model WITHOUT device_map (like Phase 1)
    print("  Loading model (this will take a while)...")
    coder_model = AutoModelForCausalLM.from_pretrained(
        coder_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"  ✓ Model loaded")

    # Manually move to GPU (like Phase 1)
    if torch.cuda.is_available():
        print("  Moving to GPU...")
        coder_model = coder_model.cuda()
        print(f"  ✓ Moved to GPU")

    coder_model = coder_model.eval()
    print()

    # Inspect the architecture
    print("  Model structure:")
    print(f"    - Model type: {type(coder_model).__name__}")
    print(f"    - Config: {type(coder_model.config).__name__}")
    print()

    # Get embedding layer
    if hasattr(coder_model, 'model'):
        # Typical structure: model.model.embed_tokens
        base_model = coder_model.model
    else:
        base_model = coder_model

    if hasattr(base_model, 'embed_tokens'):
        embed_layer = base_model.embed_tokens
    elif hasattr(base_model, 'embeddings'):
        embed_layer = base_model.embeddings
    else:
        embed_layer = None

    # Get dimensions from config and embedding layer
    config = coder_model.config

    print("  Model configuration:")
    if hasattr(config, 'hidden_size'):
        print(f"    - hidden_size: {config.hidden_size}")
    if hasattr(config, 'd_model'):
        print(f"    - d_model: {config.d_model}")
    if hasattr(config, 'n_embd'):
        print(f"    - n_embd: {config.n_embd}")
    if hasattr(config, 'vocab_size'):
        print(f"    - vocab_size: {config.vocab_size}")
    print()

    if embed_layer is not None:
        print("  Embedding layer:")
        print(f"    - Type: {type(embed_layer).__name__}")
        print(f"    - Weight shape: {embed_layer.weight.shape}")

        # embedding dimension is typically [vocab_size, embed_dim]
        coder_embed_dim = embed_layer.weight.shape[1]

        print()
        print("  " + "─" * 66)
        print(f"  Code Model Embedding Dimension: {coder_embed_dim}")
        print("  " + "─" * 66)
        print()
    else:
        print("  ⚠️  Could not find embedding layer")
        coder_embed_dim = config.hidden_size if hasattr(config, 'hidden_size') else None
        if coder_embed_dim:
            print(f"  Using hidden_size from config: {coder_embed_dim}")
            print()

    # Test with actual input
    print("  Testing with actual input...")
    test_text = "def hello_world():\n    print('Hello, world!')"
    inputs = tokenizer(test_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        # Get embeddings
        input_ids = inputs['input_ids']
        if embed_layer is not None:
            embeddings = embed_layer(input_ids)
            print(f"    Input token IDs shape: {input_ids.shape}")
            print(f"    Embeddings shape: {embeddings.shape}")

            actual_embed_dim = embeddings.shape[-1]
            print()
            print("  " + "─" * 66)
            print(f"  Verified Embedding Dimension: {actual_embed_dim}")
            print("  " + "─" * 66)

            if actual_embed_dim == coder_embed_dim:
                print("  ✅ Dimension matches config")
            else:
                print(f"  ⚠️  Mismatch: config says {coder_embed_dim}, actual is {actual_embed_dim}")
                coder_embed_dim = actual_embed_dim

    print()

    # Clean up
    del coder_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    coder_success = True

except Exception as e:
    print(f"  ❌ Error loading coder model: {e}")
    import traceback
    traceback.print_exc()
    print()
    coder_embed_dim = None
    coder_success = False


# ═══════════════════════════════════════════════════════════════════════════
# Summary & Next Steps
# ═══════════════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("  📊 SUMMARY: Embedding Dimensions")
print("=" * 70)
print()

if vision_success and vision_embed_dim:
    print(f"  Vision Encoder Output:  {vision_embed_dim:>6d} dimensions")
    status_vision = "✅"
else:
    print(f"  Vision Encoder Output:  {'FAILED':>6s}")
    status_vision = "❌"

if coder_success and coder_embed_dim:
    print(f"  Code Model Input:       {coder_embed_dim:>6d} dimensions")
    status_coder = "✅"
else:
    print(f"  Code Model Input:       {'FAILED':>6s}")
    status_coder = "❌"

print()

if vision_success and coder_success and vision_embed_dim and coder_embed_dim:
    print("  " + "─" * 66)
    print("  PROJECTION ADAPTER DESIGN:")
    print("  " + "─" * 66)
    print()
    print(f"  The adapter must map from {vision_embed_dim}D → {coder_embed_dim}D")
    print()

    # Suggest architecture
    intermediate_dim = max(vision_embed_dim, coder_embed_dim) * 2

    print("  Suggested architecture (2-layer MLP):")
    print()
    print("    ```python")
    print("    projector = nn.Sequential(")
    print(f"        nn.Linear({vision_embed_dim}, {intermediate_dim}),")
    print("        nn.GELU(),")
    print(f"        nn.Linear({intermediate_dim}, {coder_embed_dim})")
    print("    )")
    print("    ```")
    print()

    # Calculate parameter count
    params_layer1 = vision_embed_dim * intermediate_dim + intermediate_dim
    params_layer2 = intermediate_dim * coder_embed_dim + coder_embed_dim
    total_params = params_layer1 + params_layer2

    print(f"  Estimated parameters: ~{total_params / 1e6:.1f}M")
    print()

    print("  " + "─" * 66)
    print("  ✅ Ready to proceed to Phase 2: Projector training")
    print("  " + "─" * 66)

elif status_vision == "✅" and status_coder == "❌":
    print("  ⚠️  Vision encoder OK, but failed to load coder model")
    print("     This might be due to:")
    print("     - Insufficient GPU memory (need ~32GB for 16B model)")
    print("     - Model not available on HuggingFace")
    print("     - Missing dependencies")

elif status_vision == "❌" and status_coder == "✅":
    print("  ⚠️  Coder model OK, but failed to load vision encoder")

else:
    print("  ❌ Failed to load both models. Check error messages above.")

print()
print("=" * 70)
print("Done!")
print()
