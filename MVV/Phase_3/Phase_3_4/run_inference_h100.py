"""
Phase 3.4 -- Stage 1 Inference (H100, bfloat16)

Loads the Stage 1 checkpoint (ConvRoPEProjector + LoRA LLM) trained in native
bfloat16 on H100 and runs greedy generation on random code-image tensors:

    SigLIP tensor [1024, 1152]
      -> ConvRoPEProjector [256, 2048]
      -> splice into token embeddings at placeholder positions
      -> DeepSeek-Coder-V2-Lite-Instruct (bfloat16 + LoRA)
      -> greedy decode -> generated source text

Usage:
    python run_inference_h100.py
    python run_inference_h100.py --ckpt-dir /path/to/checkpoint --num-samples 5
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import random
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
# Constants -- must match train_stage1.py exactly
# ---------------------------------------------------------------------------
N_VISUAL_TOKENS = 256
MAX_GEN_TOKENS = 2048
SEED = 42


# ---------------------------------------------------------------------------
# ConvRoPEProjector -- identical to train_stage1.py
# ---------------------------------------------------------------------------
def _sinusoidal_freqs(seq_len: int, half_dim: int, device: torch.device):
    """Return (cos, sin) tables each [seq_len, half_dim]."""
    i      = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta  = 1.0 / (10000.0 ** (i / half_dim))
    pos    = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, theta)
    return angles.cos(), angles.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation. x: [..., seq_len, 2*half_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_2d_rope_16x16(x: torch.Tensor) -> torch.Tensor:
    """Apply 2D RoPE on post-convolution 16x16 grid (256 tokens)."""
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"
    half_D   = D // 2
    half_dim = half_D // 2
    device   = x.device
    rows = torch.arange(256, device=device) // 16
    cols = torch.arange(256, device=device) % 16
    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
    cos_table = cos_table.to(x.dtype)
    sin_table = sin_table.to(x.dtype)
    cos_row = cos_table[rows]
    sin_row = sin_table[rows]
    cos_col = cos_table[cols]
    sin_col = sin_table[cols]
    x_row = x[:, :, :half_D]
    x_col = x[:, :, half_D:]
    x_row_rot = _rope_rotate(x_row, cos_row, sin_row)
    x_col_rot = _rope_rotate(x_col, cos_col, sin_col)
    return torch.cat([x_row_rot, x_col_rot], dim=-1)


class ConvRoPEProjector(nn.Module):
    """Compresses [B, 1024, 1152] raw SigLIP tokens to [B, 256, 2048]."""
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
        B, N, C = x.shape
        assert N == self.grid_in ** 2 and C == self.feat_dim
        # dtype fix: cast input to match conv weights (critical for bfloat16)
        x = x.to(self.conv.weight.dtype)
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = apply_2d_rope_16x16(x)
        # ensure consistent dtype after RoPE (which uses float32 internally)
        x = x.to(self.conv.weight.dtype)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Load model for inference (bfloat16, NO quantization)
# ---------------------------------------------------------------------------
def load_model_for_inference(model_path: str, ckpt_dir: str, device: str):
    """Load tokenizer, bfloat16 LLM with LoRA weights, and ConvRoPEProjector."""
    ckpt_dir = Path(ckpt_dir)

    print(f"[load] Tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Native bfloat16 -- NO quantization (H100 has 80GB VRAM)
    print(f"[load] LLM in bfloat16 from {model_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load trained LoRA adapter
    lora_path = ckpt_dir / "lora_adapter"
    if lora_path.exists():
        print(f"[load] Loading LoRA adapter from {lora_path}")
        llm = PeftModel.from_pretrained(llm, str(lora_path), is_trainable=False)
        print("[load] LoRA adapter loaded.")
    else:
        print(f"[load] WARNING: No LoRA adapter found at {lora_path}, using base model.")

    # Load projector
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048)
    proj_path = ckpt_dir / "projector.pth"
    if proj_path.exists():
        print(f"[load] Loading projector from {proj_path}")
        ckpt = torch.load(str(proj_path), map_location="cpu", weights_only=True)
        if "projector_state_dict" in ckpt:
            projector.load_state_dict(ckpt["projector_state_dict"])
        else:
            projector.load_state_dict(ckpt)
        print("[load] Projector weights loaded.")
    else:
        print(f"[load] WARNING: No projector checkpoint at {proj_path}, using random init.")

    # Cast projector to bfloat16 and move to device
    projector = projector.to(device=device, dtype=torch.bfloat16)
    projector.eval()
    llm.eval()

    return projector, llm, tokenizer


# ---------------------------------------------------------------------------
# Generation -- matches training embedding construction exactly
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(projector, llm, tokenizer, vision_tensor, device, max_new_tokens=MAX_GEN_TOKENS):
    """
    Greedy autoregressive generation.

    Input layout matches training exactly:
        [pad_id * 256 (visual placeholders)] [newline] [generated tokens...]

    Embedding construction:
        1. Build input_ids with pad_token_id placeholders for visual positions
        2. Get text embeddings under torch.no_grad()
        3. Replace first 256 positions with projected visual embeddings
        4. Generate with inputs_embeds (NOT input_ids)
    """
    # Project vision features -- keep in bfloat16
    vision = vision_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, 1024, 1152]
    visual_embeds = projector(vision)  # [1, 256, 2048]

    # Build initial input: 256 placeholder tokens + newline
    pad_id = tokenizer.pad_token_id
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    input_ids = [pad_id] * N_VISUAL_TOKENS + newline_ids
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, 257]

    # Build embeddings -- torch.no_grad() prevents corrupting embedding space
    with torch.no_grad():
        text_embeds = llm.get_input_embeddings()(input_ids)
    text_embeds = text_embeds.clone()

    # Splice in visual embeddings (match dtype of text embeddings)
    visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
    text_embeds[:, :N_VISUAL_TOKENS, :] = visual_embeds

    # Initial forward pass to get KV cache
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    outputs = llm(
        inputs_embeds=text_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]

    generated_ids = [next_token.item()]
    eos_id = tokenizer.eos_token_id

    # Autoregressive loop (greedy, temperature=0)
    for step in range(max_new_tokens - 1):
        if generated_ids[-1] == eos_id:
            break

        cur_pos = seq_len + len(generated_ids) - 1
        position_ids = torch.tensor([[cur_pos]], device=device)
        attention_mask = torch.ones(1, seq_len + len(generated_ids), dtype=torch.long, device=device)

        outputs = llm(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_line_overlap(ground_truth: str, generated: str) -> float:
    """Percentage of non-empty ground truth lines that appear exactly in generated text."""
    gt_lines = [line.strip() for line in ground_truth.splitlines() if line.strip()]
    if not gt_lines:
        return 0.0
    gen_text = generated.strip()
    matches = sum(1 for line in gt_lines if line in gen_text)
    return 100.0 * matches / len(gt_lines)


def compute_char_accuracy(ground_truth: str, generated: str) -> float:
    """Character-level accuracy (ratio of matching chars at same positions)."""
    gt = ground_truth.rstrip()
    gen = generated.rstrip()
    if not gt:
        return 0.0
    matches = sum(1 for a, b in zip(gt, gen) if a == b)
    return 100.0 * matches / len(gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3.4 -- Stage 1 Inference (H100, bfloat16)")
    parser.add_argument("--model-path", type=str,
                        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--ckpt-dir", type=str,
                        default="MVV/Phase_3/checkpoints/stage1_4h100/epoch_best",
                        help="Checkpoint directory (must contain projector.pth and lora_adapter/)")
    parser.add_argument("--data-dir", type=str,
                        default="MVV/Phase_3/full_data/tensors_and_texts",
                        help="Directory containing paired .pt and .txt files")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples to evaluate")
    parser.add_argument("--max-tokens", type=int, default=MAX_GEN_TOKENS,
                        help="Maximum tokens to generate per sample")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Phase 3.4 -- Stage 1 Inference (H100, bfloat16)")
    print("=" * 80)
    print(f"  Device:      {device}")
    if device == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Model:       {args.model_path}")
    print(f"  Checkpoint:  {args.ckpt_dir}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Dtype:       bfloat16 (native, no quantization)")
    print()

    # Discover paired samples
    data_dir = Path(args.data_dir)
    pt_files = {p.stem: p for p in data_dir.glob("*.pt")}
    txt_files = {p.stem: p for p in data_dir.glob("*.txt")}
    common_stems = sorted(set(pt_files.keys()) & set(txt_files.keys()))

    if not common_stems:
        print(f"ERROR: No paired .pt/.txt files found in {data_dir}")
        return

    print(f"[data] Found {len(common_stems)} paired samples in {data_dir}")

    # Pick random samples
    random.seed(args.seed)
    selected = random.sample(common_stems, min(args.num_samples, len(common_stems)))
    print(f"[data] Selected samples: {selected}")
    print()

    # Load model
    projector, llm, tokenizer = load_model_for_inference(args.model_path, args.ckpt_dir, device)

    vram_mb = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
    print(f"[load] VRAM after model load: {vram_mb:.0f} MB")
    print()

    # Run inference on each sample
    all_line_overlaps = []
    all_char_accs = []

    for i, stem in enumerate(selected, 1):
        pt_path = pt_files[stem]
        txt_path = txt_files[stem]

        print("=" * 80)
        print(f"SAMPLE {i}/{len(selected)}: {stem}")
        print("=" * 80)

        # Load vision tensor
        vision_tensor = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        assert vision_tensor.shape == (1024, 1152), f"Bad shape: {vision_tensor.shape}"

        # Load ground truth
        ground_truth = txt_path.read_text(encoding="utf-8")
        gt_lines = ground_truth.splitlines()

        print(f"  Vision tensor: {vision_tensor.shape} ({vision_tensor.dtype})")
        print(f"  Ground truth:  {len(gt_lines)} lines, {len(ground_truth)} chars")
        print()

        # Generate
        print("  Generating (greedy, temperature=0)...")
        generated = generate(projector, llm, tokenizer, vision_tensor, device,
                             max_new_tokens=args.max_tokens)
        gen_lines = generated.splitlines()

        # Display ground truth (truncated)
        print()
        print("-" * 40)
        print("GROUND TRUTH (first 30 lines):")
        print("-" * 40)
        for line in gt_lines[:30]:
            print(line)
        if len(gt_lines) > 30:
            print(f"... ({len(gt_lines) - 30} more lines)")

        # Display generated output
        print()
        print("-" * 40)
        print("MODEL OUTPUT (first 50 lines):")
        print("-" * 40)
        for line in gen_lines[:50]:
            print(line)
        if len(gen_lines) > 50:
            print(f"... ({len(gen_lines) - 50} more lines)")

        # Metrics
        line_overlap = compute_line_overlap(ground_truth, generated)
        char_acc = compute_char_accuracy(ground_truth, generated)
        all_line_overlaps.append(line_overlap)
        all_char_accs.append(char_acc)

        print()
        print("-" * 40)
        print("METRICS:")
        print("-" * 40)
        print(f"  Line overlap:    {line_overlap:.1f}% of ground truth lines found in output")
        print(f"  Char accuracy:   {char_acc:.1f}% (positional character match)")
        print(f"  Generated:       {len(generated)} chars, {len(gen_lines)} lines")
        print(f"  Ground truth:    {len(ground_truth)} chars, {len(gt_lines)} lines")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Samples evaluated: {len(selected)}")
    if all_line_overlaps:
        avg_overlap = sum(all_line_overlaps) / len(all_line_overlaps)
        avg_char = sum(all_char_accs) / len(all_char_accs)
        print(f"  Avg line overlap:  {avg_overlap:.1f}%")
        print(f"  Avg char accuracy: {avg_char:.1f}%")
        print()
        print("  Per-sample breakdown:")
        for i, stem in enumerate(selected):
            print(f"    [{i+1}] {stem}: line={all_line_overlaps[i]:.1f}%, char={all_char_accs[i]:.1f}%")
    print()


if __name__ == "__main__":
    main()
