#!/usr/bin/env python3
"""
Phase 3.2 — Architecture Plumbing Test
=======================================
Verifies that QLoRA freezes the 16B backbone correctly, that LoRA adapters
and the ConvRoPEProjector remain trainable, and that gradients flow from the
language-model loss all the way back through both LoRA and the projector.

Entirely self-contained: no imports from other project files.
"""

import os
import sys

# Ensure we don't pick up stale user-site packages (e.g. transformers 5.x)
os.environ["PYTHONNOUSERSITE"] = "1"

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# ConvRoPEProjector (copied verbatim from Phase 3)
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
# CoderVLModel wrapper (same splice logic as Phase 3)
# ---------------------------------------------------------------------------

N_VISUAL_TOKENS = 256


class CoderVLModel(nn.Module):
    """Wraps ConvRoPEProjector + LoRA-adapted DeepSeek LLM."""
    def __init__(self, projector: ConvRoPEProjector, llm: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.llm = llm

    def forward(self, vision, input_ids, attention_mask, labels):
        B, S = input_ids.shape
        visual_embeds = self.projector(vision)
        text_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
        text_embeds[:, :N_VISUAL_TOKENS, :] = visual_embeds
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs.loss


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = str(
        Path.home()
        / ".cache/huggingface/hub"
        / "models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct"
        / "snapshots"
        / "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"
    )
    print(f"Model path: {model_path}")

    # ------------------------------------------------------------------
    # Step 1: Load tokenizer
    # ------------------------------------------------------------------
    print("\n--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Step 2: Load LLM with 8-bit quantization
    # ------------------------------------------------------------------
    print("\n--- Loading LLM (8-bit) ---")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    llm = prepare_model_for_kbit_training(llm)

    # ------------------------------------------------------------------
    # Step 3: Apply LoRA
    # ------------------------------------------------------------------
    print("\n--- Applying LoRA ---")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Step 4: Create projector (random init, no checkpoint)
    # ------------------------------------------------------------------
    print("\n--- Creating ConvRoPEProjector ---")
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048).to(device).float()

    # ------------------------------------------------------------------
    # Step 5: Create CoderVLModel wrapper
    # ------------------------------------------------------------------
    model = CoderVLModel(projector, llm)

    # ------------------------------------------------------------------
    # Step 6: Setup optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW([
        {"params": model.projector.parameters(), "lr": 1e-5},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": 5e-6},
    ])

    # ------------------------------------------------------------------
    # Step 7: Create dummy inputs
    # ------------------------------------------------------------------
    print("\n--- Creating dummy inputs ---")
    vision = torch.randn(1, 1024, 1152, device=device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    # 256 pad tokens + 50 random tokens + 1 EOS = 307 total
    random_ids = torch.randint(100, 5000, (50,))
    input_ids = torch.cat([
        torch.full((256,), pad_token_id, dtype=torch.long),
        random_ids,
        torch.tensor([eos_token_id], dtype=torch.long),
    ]).unsqueeze(0).to(device)  # [1, 307]

    seq_len = input_ids.shape[1]
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    # Labels: -100 for the first 256 (visual) positions, real IDs for the rest
    labels = torch.cat([
        torch.full((256,), -100, dtype=torch.long),
        random_ids,
        torch.tensor([eos_token_id], dtype=torch.long),
    ]).unsqueeze(0).to(device)  # [1, 307]

    print(f"  vision shape:         {vision.shape}")
    print(f"  input_ids shape:      {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")
    print(f"  labels shape:         {labels.shape}")

    # ------------------------------------------------------------------
    # Step 8: Run checks
    # ------------------------------------------------------------------
    results = {}

    # --- Check 1: Parameter Freeze Verification ---
    print("\n" + "=" * 60)
    print("CHECK 1: Parameter Freeze Verification")
    print("=" * 60)

    proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    lora_trainable = sum(
        p.numel() for n, p in model.llm.named_parameters() if p.requires_grad
    )
    frozen_params = sum(
        p.numel() for n, p in model.llm.named_parameters() if not p.requires_grad
    )

    print(f"  Projector trainable params:  {proj_trainable:,}")
    print(f"  LoRA trainable params:       {lora_trainable:,}")
    print(f"  Frozen backbone params:      {frozen_params:,}")

    check1 = (
        proj_trainable > 0
        and lora_trainable > 0
        and frozen_params > 0
        and frozen_params > (proj_trainable + lora_trainable)
    )
    results["Check 1: Param Freeze"] = check1
    print(f"  --> {'PASS' if check1 else 'FAIL'}")

    # --- Check 2: requires_grad correctness ---
    print("\n" + "=" * 60)
    print("CHECK 2: requires_grad Correctness")
    print("=" * 60)

    proj_all_grad = all(p.requires_grad for p in model.projector.parameters())
    lora_grad_ok = any(p.requires_grad for n, p in model.llm.named_parameters())
    frozen_grad_ok = any(not p.requires_grad for n, p in model.llm.named_parameters())

    print(f"  All projector params requires_grad=True: {proj_all_grad}")
    print(f"  At least one LoRA param requires_grad=True: {lora_grad_ok}")
    print(f"  At least one backbone param requires_grad=False: {frozen_grad_ok}")

    check2 = proj_all_grad and lora_grad_ok and frozen_grad_ok
    results["Check 2: requires_grad"] = check2
    print(f"  --> {'PASS' if check2 else 'FAIL'}")

    # --- Check 3: Forward pass ---
    print("\n" + "=" * 60)
    print("CHECK 3: Forward Pass")
    print("=" * 60)

    try:
        loss = model(vision, input_ids, attention_mask, labels)
        loss_val = loss.item()
        loss_ok = loss.dim() == 0 and loss.requires_grad and torch.isfinite(loss) and loss_val > 0
        print(f"  Loss value:        {loss_val:.4f}")
        print(f"  Loss is scalar:    {loss.dim() == 0}")
        print(f"  Loss requires_grad: {loss.requires_grad}")
        print(f"  Loss is finite:    {torch.isfinite(loss).item()}")
        print(f"  Loss > 0:          {loss_val > 0}")
        check3 = loss_ok
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        check3 = False
        loss = None

    results["Check 3: Forward"] = check3
    print(f"  --> {'PASS' if check3 else 'FAIL'}")

    # --- Check 4: Backward pass & gradient flow ---
    print("\n" + "=" * 60)
    print("CHECK 4: Backward Pass & Gradient Flow")
    print("=" * 60)

    check4_parts = {}
    if loss is not None and check3:
        try:
            # Save initial conv weights for Check 5
            conv_weight_before = model.projector.conv.weight.data.clone()

            loss.backward()

            # Projector gradients
            conv_grad = model.projector.conv.weight.grad is not None
            mlp0_grad = model.projector.mlp[0].weight.grad is not None
            mlp2_grad = model.projector.mlp[2].weight.grad is not None

            check4_parts["projector.conv.weight grad"] = conv_grad
            check4_parts["projector.mlp[0].weight grad"] = mlp0_grad
            check4_parts["projector.mlp[2].weight grad"] = mlp2_grad

            print(f"  projector.conv.weight.grad is not None: {conv_grad}")
            print(f"  projector.mlp[0].weight.grad is not None: {mlp0_grad}")
            print(f"  projector.mlp[2].weight.grad is not None: {mlp2_grad}")

            # LoRA gradients
            lora_has_grad = False
            first_lora_grad_name = None
            first_lora_grad_norm = None
            for n, p in model.llm.named_parameters():
                if p.requires_grad and p.grad is not None:
                    lora_has_grad = True
                    first_lora_grad_name = n
                    first_lora_grad_norm = p.grad.norm().item()
                    break

            check4_parts["at least one LoRA param has grad"] = lora_has_grad
            print(f"  At least one LoRA param has .grad: {lora_has_grad}")

            # Print grad norms
            print(f"\n  Grad norms:")
            if conv_grad:
                print(f"    projector.conv.weight:  {model.projector.conv.weight.grad.norm().item():.6e}")
            if mlp0_grad:
                print(f"    projector.mlp[0].weight: {model.projector.mlp[0].weight.grad.norm().item():.6e}")
            if mlp2_grad:
                print(f"    projector.mlp[2].weight: {model.projector.mlp[2].weight.grad.norm().item():.6e}")
            if lora_has_grad:
                print(f"    LoRA ({first_lora_grad_name}): {first_lora_grad_norm:.6e}")

            check4 = all(check4_parts.values())
        except Exception as e:
            print(f"  EXCEPTION during backward: {e}")
            import traceback
            traceback.print_exc()
            check4 = False
            conv_weight_before = None
    else:
        print("  SKIPPED (forward pass failed)")
        check4 = False
        conv_weight_before = None

    results["Check 4: Backward"] = check4
    print(f"  --> {'PASS' if check4 else 'FAIL'}")

    # --- Check 5: Optimizer step ---
    print("\n" + "=" * 60)
    print("CHECK 5: Optimizer Step")
    print("=" * 60)

    if check4 and conv_weight_before is not None:
        try:
            optimizer.step()
            conv_weight_after = model.projector.conv.weight.data.clone()
            weight_changed = not torch.equal(conv_weight_before, conv_weight_after)
            print(f"  projector.conv.weight changed after step: {weight_changed}")
            check5 = weight_changed
        except Exception as e:
            print(f"  EXCEPTION during optimizer step: {e}")
            check5 = False
    else:
        print("  SKIPPED (backward pass failed)")
        check5 = False

    results["Check 5: Optimizer"] = check5
    print(f"  --> {'PASS' if check5 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"FAILED CHECKS: {', '.join(failed)}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
