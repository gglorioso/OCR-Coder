"""
Test 1: Perfect Features Experiment

Replace visual features with text embeddings from the coder model itself.
This tests if the token insertion mechanism works when features are in the
correct representation space.

If this works → projection adapter is the bottleneck
If this fails → token insertion mechanism is broken
"""

import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import json

from projector import ProjectionAdapter


class PerfectFeaturesDataset(Dataset):
    """
    Dataset that uses text embeddings as "visual" features.

    For each example, we tokenize the actual code content (ground truth)
    and use its embeddings as the "visual" features. This gives the model
    perfect information in the correct representation space.
    """

    def __init__(self, manifest_path, tokenizer, max_seq_length=2048, num_visual_tokens=256):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_visual_tokens = num_visual_tokens

        # Load manifest
        with open(manifest_path) as f:
            self.examples = [json.loads(line) for line in f]

        print(f"  Loaded {len(self.examples)} examples from {manifest_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format: <img_start><image><img_end>\n{question}\n\n{answer}
        # We'll extract the actual code content to create "perfect" visual features
        conversation = ex["conversations"]
        question = conversation[0]["content"]  # Contains <img_start><image><img_end>\n{actual_question}
        answer = conversation[1]["content"]

        # For perfect features, we need the actual CODE content
        # In our dataset, the code is what should be "visible" in the image
        # For simplicity, we'll use the answer text as the "visual" content
        # (In reality, we'd extract the code from the image path, but this is a test)
        visual_text = answer[:500]  # Use first 500 chars of answer as "code content"

        # Tokenize the full conversation (question + answer) for training
        full_text = question + "\n\n" + answer
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)

        # Create labels (mask out question, keep answer)
        labels = input_ids.clone()

        # Find where the answer starts (after "\n\n")
        # For now, we'll mask out everything before <image> token and the question
        # This is a simplification - in reality we'd need to find the exact split
        question_only = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        labels[:len(question_only)] = -100  # Mask question

        # Tokenize the visual content (this will be converted to embeddings)
        visual_encoded = self.tokenizer(
            visual_text,
            max_length=self.num_visual_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        visual_token_ids = visual_encoded["input_ids"].squeeze(0)  # [num_visual_tokens]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "visual_token_ids": visual_token_ids,  # Will be converted to embeddings
        }


def collate_fn(batch):
    """Custom collate to pad sequences."""
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = []
    labels = []
    visual_token_ids = []

    for item in batch:
        # Pad input_ids
        ids = item["input_ids"]
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        input_ids.append(ids)

        # Pad labels
        lbl = item["labels"]
        if pad_len > 0:
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
        labels.append(lbl)

        # Visual token IDs (already padded to fixed length)
        visual_token_ids.append(item["visual_token_ids"])

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "visual_token_ids": torch.stack(visual_token_ids),
    }


def replace_image_tokens_perfect(input_ids, visual_embeds, image_token_id, text_embed_fn):
    """
    Replace <image> tokens with "perfect" visual embeddings (derived from text).

    Args:
        input_ids: [B, seq] - token IDs containing <image>
        visual_embeds: [B, vis_tok, dim] - embeddings from tokenized code content
        image_token_id: int
        text_embed_fn: embedding layer

    Returns:
        combined_embeds: [B, new_seq, dim]
        attention_mask: [B, new_seq]
    """
    batch = input_ids.size(0)
    text_embeds = text_embed_fn(input_ids)  # [B, seq, dim]

    out_embeds = []
    for i in range(batch):
        positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            out_embeds.append(text_embeds[i])
        else:
            p = positions[0].item()
            combined = torch.cat(
                [text_embeds[i, :p], visual_embeds[i], text_embeds[i, p + 1:]],
                dim=0,
            )
            out_embeds.append(combined)

    # Pad to longest sequence
    max_len = max(e.size(0) for e in out_embeds)
    padded, masks = [], []
    for e in out_embeds:
        pad = max_len - e.size(0)
        if pad > 0:
            e = torch.cat([e, torch.zeros(pad, e.size(1), device=e.device, dtype=e.dtype)])
        mask = torch.ones(max_len, device=e.device)
        if pad > 0:
            mask[-pad:] = 0
        padded.append(e)
        masks.append(mask)

    return torch.stack(padded), torch.stack(masks)


def expand_labels_perfect(labels, input_ids, image_token_id, num_visual_tokens):
    """Expand labels to account for visual tokens."""
    batch = labels.size(0)
    out = []
    for i in range(batch):
        positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            out.append(labels[i])
        else:
            p = positions[0].item()
            vis = torch.full((num_visual_tokens,), -100, dtype=labels.dtype, device=labels.device)
            out.append(torch.cat([labels[i, :p], vis, labels[i, p + 1:]]))

    max_len = max(l.size(0) for l in out)
    padded = []
    for l in out:
        pad = max_len - l.size(0)
        if pad > 0:
            l = torch.cat([l, torch.full((pad,), -100, dtype=l.dtype, device=l.device)])
        padded.append(l)

    return torch.stack(padded)


@torch.no_grad()
def evaluate(adapter, coder, loader, image_token_id, embed_fn, device):
    """Evaluate on validation set."""
    adapter.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        vis_ids = batch["visual_token_ids"].to(device)

        # Get "perfect" visual embeddings from text
        visual_embeds = embed_fn(vis_ids).half()  # [B, num_visual_tokens, dim]

        # Pass through adapter (should learn identity or simple transform)
        projected = adapter(visual_embeds).half()

        # Replace <image> tokens
        embeds, mask = replace_image_tokens_perfect(ids, projected, image_token_id, embed_fn)
        adj_labels = expand_labels_perfect(lbl, ids, image_token_id, vis_ids.size(1))

        loss = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def save_checkpoint(adapter, optimizer, scheduler, step, epoch, ckpt_dir, name):
    """Save training checkpoint."""
    path = os.path.join(ckpt_dir, f"{name}.pt")
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step,
        "epoch": epoch,
    }, path)
    print(f"  Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test 1: Perfect Features Experiment")
    parser.add_argument("--train_manifest", default="Data Crawling/output/manifests/train.jsonl")
    parser.add_argument("--val_manifest", default="Data Crawling/output/manifests/val.jsonl")
    parser.add_argument("--coder_model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_visual_tokens", type=int, default=256)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/perfect_features")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = "cuda"

    print("=" * 70)
    print("TEST 1: PERFECT FEATURES EXPERIMENT")
    print("=" * 70)
    print("Using text embeddings as 'visual' features to test if token")
    print("insertion mechanism works when features are in correct space.")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. Load coder model (4-bit quantized, frozen)
    # ========================================================================
    print("Loading coder model (4-bit, fp16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.coder_model, trust_remote_code=True)

    # Add special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze coder
    for p in coder.parameters():
        p.requires_grad = False

    coder.gradient_checkpointing_enable()
    coder.train()

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    coder_dim = coder.config.hidden_size
    print(f"  hidden_size={coder_dim}, image_token_id={image_token_id}")
    print(f"  Coder model loaded and frozen ✓\n")

    # ========================================================================
    # 2. Create adapter (should learn identity mapping)
    # ========================================================================
    print(f"Creating adapter ({coder_dim} -> {coder_dim}) ...")
    print("  NOTE: Input is already in coder space, adapter should learn identity")
    adapter = ProjectionAdapter(vision_dim=coder_dim, hidden_dim=4096, coder_dim=coder_dim)
    adapter = adapter.to(device)
    print(f"  Parameters: {adapter.num_parameters():,}\n")

    # ========================================================================
    # 3. Load datasets
    # ========================================================================
    print("Loading datasets...")
    train_ds = PerfectFeaturesDataset(
        args.train_manifest, tokenizer, args.max_seq_length, args.num_visual_tokens
    )
    val_ds = PerfectFeaturesDataset(
        args.val_manifest, tokenizer, args.max_seq_length, args.num_visual_tokens
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(f"  Train: {len(train_ds)} examples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} examples, {len(val_loader)} batches\n")

    # ========================================================================
    # 4. Training setup
    # ========================================================================
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    embed_fn = coder.get_input_embeddings()

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        adapter.train()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            ids = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            vis_ids = batch["visual_token_ids"].to(device)

            # Get "perfect" visual embeddings from text tokens
            visual_embeds = embed_fn(vis_ids).half()  # [B, num_visual_tokens, dim]

            # Pass through adapter
            projected = adapter(visual_embeds).half()

            # Replace <image> tokens
            embeds, mask = replace_image_tokens_perfect(ids, projected, image_token_id, embed_fn)
            adj_labels = expand_labels_perfect(lbl, ids, image_token_id, vis_ids.size(1))

            # Forward pass
            outputs = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    print(f"  Step {global_step}: loss={loss.item() * args.grad_accum:.4f}")

                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(adapter, coder, val_loader, image_token_id, embed_fn, device)
                    print(f"  Step {global_step}: val_loss={val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(adapter, optimizer, scheduler, global_step, epoch,
                                      args.checkpoint_dir, "best")

                    adapter.train()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print()
    print("INTERPRETATION:")
    print("  - If val_loss is LOW (~1.0-1.5): Token insertion works! Problem is projection.")
    print("  - If val_loss is HIGH (~3.0+): Token insertion mechanism is broken.")
    print("=" * 70)


if __name__ == "__main__":
    main()
