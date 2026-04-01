"""
Phase 3.3 -- Inference Demo

Loads the Phase 3.3 checkpoint (ConvRoPEProjector + LoRA LLM) and runs
greedy generation on a single code-image tensor to demonstrate the
end-to-end pipeline:

    SigLIP tensor [1024, 1152]
      -> ConvRoPEProjector [256, 2048]
      -> splice into token embeddings
      -> DeepSeek-Coder-V2-Lite-Instruct (8-bit + LoRA)
      -> greedy decode -> generated source text

Usage:
    python run_inference.py --model-path /path/to/DeepSeek-Coder-V2-Lite-Instruct
    python run_inference.py --model-path /path/to/model --sample black__docs__conf_py_chunk0
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ---------------------------------------------------------------------------
# Constants — must match train_joint.py exactly
# ---------------------------------------------------------------------------
MAX_TEXT_TOKENS = 768
N_VISUAL_TOKENS = 256
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_GEN_TOKENS = 512

DEFAULT_SAMPLE = "black__action__main_py_chunk0"
DEFAULT_DATA_DIR = "MVV/Phase_3/full_data/tensors_and_texts"
DEFAULT_CKPT_DIR = "MVV/Phase_3/Phase_3_3/checkpoints/epoch_9"


# ---------------------------------------------------------------------------
# ConvRoPEProjector — identical to train_joint.py
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
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"
    half_D   = D // 2
    half_dim = half_D // 2
    device   = x.device
    rows = torch.arange(256, device=device) // 16
    cols = torch.arange(256, device=device) % 16
    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
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
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = apply_2d_rope_16x16(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Load model for inference
# ---------------------------------------------------------------------------
def load_model_for_inference(model_path: str, ckpt_dir: str, device: str):
    """Load tokenizer, 8-bit LLM with LoRA weights, and ConvRoPEProjector from checkpoint."""
    ckpt_dir = Path(ckpt_dir)

    print(f"[load] Tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] LLM in 8-bit from {model_path}")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    llm = prepare_model_for_kbit_training(llm)

    # Apply LoRA structure (same config as training)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)

    # Load trained LoRA weights
    lora_path = ckpt_dir / "lora_adapter"
    if lora_path.exists():
        print(f"[load] Loading LoRA weights from {lora_path}")
        llm.load_adapter(str(lora_path), adapter_name="default", is_trainable=False)
        print("[load] LoRA weights loaded.")
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

    projector = projector.to(device)
    projector.eval()
    llm.eval()

    return projector, llm, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(projector, llm, tokenizer, vision_tensor, device, max_new_tokens=MAX_GEN_TOKENS):
    """
    Greedy autoregressive generation.

    Input layout matches training:
        [visual_placeholders (256)] [newline] [generated tokens...]
    """
    # Project vision features
    vision = vision_tensor.unsqueeze(0).to(device).float()  # [1, 1024, 1152]
    visual_embeds = projector(vision)  # [1, 256, 2048]

    # Build initial input: 256 placeholder tokens + newline
    pad_id = tokenizer.pad_token_id
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    input_ids = [pad_id] * N_VISUAL_TOKENS + newline_ids
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, 257]

    # Build initial embeddings with vision splice
    text_embeds = llm.get_input_embeddings()(input_ids).clone()
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

    # Autoregressive loop
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
# Overlap metric
# ---------------------------------------------------------------------------
def compute_line_overlap(ground_truth: str, generated: str) -> float:
    """Percentage of non-empty ground truth lines that appear in generated text."""
    gt_lines = [line.strip() for line in ground_truth.splitlines() if line.strip()]
    if not gt_lines:
        return 0.0
    gen_text = generated.strip()
    matches = sum(1 for line in gt_lines if line in gen_text)
    return 100.0 * matches / len(gt_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3.3 -- Inference Demo")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to DeepSeek-Coder-V2-Lite-Instruct model")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR,
                        help="Path to checkpoint directory (must contain projector.pth and lora_adapter/)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing .pt and .txt sample files")
    parser.add_argument("--sample", type=str, default=DEFAULT_SAMPLE,
                        help="Sample stem name (without .pt/.txt extension)")
    parser.add_argument("--max-tokens", type=int, default=MAX_GEN_TOKENS,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Device: {device}")
    print(f"[main] Model:  {args.model_path}")
    print(f"[main] Checkpoint: {args.ckpt_dir}")
    print(f"[main] Sample: {args.sample}")
    print()

    # Resolve paths
    data_dir = Path(args.data_dir)
    pt_path = data_dir / f"{args.sample}.pt"
    txt_path = data_dir / f"{args.sample}.txt"

    if not pt_path.exists():
        print(f"ERROR: Vision tensor not found: {pt_path}")
        return
    if not txt_path.exists():
        print(f"ERROR: Ground truth text not found: {txt_path}")
        return

    # Load data
    print(f"[data] Loading vision tensor: {pt_path}")
    vision_tensor = torch.load(str(pt_path), map_location="cpu", weights_only=True).float()
    print(f"[data] Vision tensor shape: {vision_tensor.shape}")
    assert vision_tensor.shape == (1024, 1152), f"Bad shape: {vision_tensor.shape}"

    print(f"[data] Loading ground truth:  {txt_path}")
    ground_truth = txt_path.read_text(encoding="utf-8")
    gt_lines = ground_truth.splitlines()
    print(f"[data] Ground truth: {len(gt_lines)} lines, {len(ground_truth)} chars")
    print()

    # Load model
    projector, llm, tokenizer = load_model_for_inference(args.model_path, args.ckpt_dir, device)
    print()

    # Generate
    print("=" * 80)
    print("GENERATING...")
    print("=" * 80)
    generated = generate(projector, llm, tokenizer, vision_tensor, device, max_new_tokens=args.max_tokens)
    print()

    # Display results
    print("=" * 80)
    print(f"SAMPLE: {args.sample}")
    print("=" * 80)

    print()
    print("-" * 40)
    print("GROUND TRUTH (first 50 lines):")
    print("-" * 40)
    for line in gt_lines[:50]:
        print(line)
    if len(gt_lines) > 50:
        print(f"... ({len(gt_lines) - 50} more lines)")

    print()
    print("-" * 40)
    print("MODEL OUTPUT:")
    print("-" * 40)
    print(generated)

    print()
    print("-" * 40)
    print("METRICS:")
    print("-" * 40)
    overlap = compute_line_overlap(ground_truth, generated)
    print(f"  Line overlap: {overlap:.1f}% of ground truth lines found in generated text")
    print(f"  Generated length: {len(generated)} chars, {len(generated.splitlines())} lines")
    print(f"  Ground truth length: {len(ground_truth)} chars, {len(gt_lines)} lines")
    print()


if __name__ == "__main__":
    main()
