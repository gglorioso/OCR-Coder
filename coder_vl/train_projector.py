"""
Phase 2a Training Script - Projection Adapter Alignment

Trains the projection adapter to map vision tokens to coder embedding space.
- Trainable: Adapter only (13.6M params)
- Frozen: Vision encoder + coder model

Usage:
    python train_projector.py --config config_phase2a.yaml
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from model import CoderVLModel


@dataclass
class TrainingConfig:
    """Phase 2a training configuration."""

    # Data
    train_manifest: str = "Data Crawling/output/manifests/train.jsonl"
    val_manifest: str = "Data Crawling/output/manifests/val.jsonl"
    image_root: str = "Data Crawling/output/images"

    # Model paths
    vision_encoder_path: str = "./models/vision_encoder.pt"
    coder_model_path: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

    # Training hyperparameters (Phase 2a - from PHASE2_PLAN.md Section 5)
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    learning_rate: float = 1e-3  # High LR safe for adapter-only
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_epochs: int = 1
    max_seq_length: int = 2048

    # Optimization
    precision: str = "bf16"  # bfloat16 for H100
    gradient_checkpointing: bool = True

    # Checkpointing and eval
    checkpoint_dir: str = "./checkpoints/phase2a"
    checkpoint_interval_minutes: int = 30
    eval_steps: int = 50
    save_total_limit: int = 3  # Keep last 3 checkpoints

    # Logging
    log_steps: int = 10
    wandb_project: str = "deepseek-coder-vl"
    wandb_run_name: str = "phase2a-adapter"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Distributed training
    distributed: bool = False  # Auto-set in main() if multiple GPUs
    local_rank: int = -1       # Set via environment variable
    world_size: int = 1        # Total number of processes


class CodeVLDataset(Dataset):
    """Dataset for code image + QA pairs."""

    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        tokenizer,
        max_seq_length: int = 2048,
    ):
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load manifest
        print(f"Loading manifest from {manifest_path}...")
        self.examples = []
        with open(manifest_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        print(f"  Loaded {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        # Load and preprocess image
        image_path = example["image"]
        # Make path absolute if relative
        if not Path(image_path).is_absolute():
            image_path = self.image_root.parent.parent / image_path

        image = Image.open(image_path).convert("RGB")
        # TODO: Apply vision encoder's preprocessing transform
        # For now, placeholder - this needs to match DeepSeek-OCR-2's preprocessing
        image_tensor = self._preprocess_image(image)

        # Build conversation text
        conversation = example["conversations"]
        user_msg = conversation[0]["content"]
        assistant_msg = conversation[1]["content"]

        # Format as instruction template
        # DeepSeek-Coder uses format: "User: {user}\n\nAssistant: {assistant}"
        full_text = f"User: {user_msg}\n\nAssistant: {assistant_msg}"

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )

        input_ids = tokenized["input_ids"].squeeze(0)  # [seq_len]

        # Create labels (mask out user prompt and image tokens)
        labels = input_ids.clone()

        # Find "Assistant:" position to mask user prompt
        assistant_token = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        assistant_start = self._find_subsequence(input_ids, assistant_token)

        if assistant_start != -1:
            # Mask everything before "Assistant:" (user prompt + image)
            labels[:assistant_start + len(assistant_token)] = -100
        else:
            # Fallback: mask first half (heuristic)
            labels[:len(labels)//2] = -100

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor,
            "example_id": example["id"],
        }

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for vision encoder.

        TODO: This should match DeepSeek-OCR-2's preprocessing exactly.
        For now, placeholder that needs to be replaced.
        """
        # Placeholder preprocessing
        # Real implementation needs to use DeepSeek-OCR-2's processor
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transform(image)

    def _find_subsequence(self, tensor: torch.Tensor, subseq: List[int]) -> int:
        """Find starting position of subsequence in tensor. Returns -1 if not found."""
        tensor_list = tensor.tolist()
        subseq_len = len(subseq)

        for i in range(len(tensor_list) - subseq_len + 1):
            if tensor_list[i:i+subseq_len] == subseq:
                return i

        return -1


class Trainer:
    """Trainer for Phase 2a adapter alignment."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.is_main_process = not config.distributed or config.local_rank == 0

        # Initialize wandb (only on main process)
        if config.wandb_project and self.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Initialize model
        print("\n" + "="*60)
        print("INITIALIZING CODERVL MODEL")
        print("="*60)

        self.model = CoderVLModel(
            vision_encoder_path=config.vision_encoder_path,
            coder_model_path=config.coder_model_path,
            freeze_vision=True,
            freeze_coder=True,
        )

        print(f"\nTrainable parameters: {self.model.num_trainable_parameters():,}")

        # Move to device
        self.model = self.model.to(config.device)

        # Wrap with DDP if distributed
        if config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                find_unused_parameters=False,  # All adapter params are used
            )
            print(f"Model wrapped with DDP (rank {config.local_rank}/{config.world_size}) ✓")

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            # Access underlying model if wrapped with DDP
            model_to_checkpoint = self.model.module if config.distributed else self.model
            model_to_checkpoint.coder_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled ✓")

        # Setup precision
        self.scaler = None
        if config.precision == "bf16":
            self.dtype = torch.bfloat16
            print(f"Using {config.precision} precision ✓")
        else:
            self.dtype = torch.float32

        # Create datasets
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)

        self.train_dataset = CodeVLDataset(
            manifest_path=config.train_manifest,
            image_root=config.image_root,
            tokenizer=self.model.tokenizer,
            max_seq_length=config.max_seq_length,
        )

        self.val_dataset = CodeVLDataset(
            manifest_path=config.val_manifest,
            image_root=config.image_root,
            tokenizer=self.model.tokenizer,
            max_seq_length=config.max_seq_length,
        )

        # Create dataloaders with optional distributed sampler
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=config.world_size,
            rank=config.local_rank,
            shuffle=True,
        ) if config.distributed else None

        val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=config.world_size,
            rank=config.local_rank,
            shuffle=False,
        ) if config.distributed else None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size_per_gpu,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size_per_gpu,
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.train_sampler = train_sampler  # Save for epoch shuffling

        # Setup optimizer (handle DDP wrapper)
        model_unwrapped = self.model.module if config.distributed else self.model
        self.optimizer = AdamW(
            model_unwrapped.adapter.parameters(),  # Only adapter is trainable
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup learning rate scheduler
        num_training_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        print(f"\nTraining steps: {num_training_steps:,}")
        print(f"Warmup steps: {num_warmup_steps:,}")
        effective_batch = config.batch_size_per_gpu * config.gradient_accumulation_steps * config.world_size
        print(f"Effective batch size: {effective_batch} (per_gpu={config.batch_size_per_gpu}, accum={config.gradient_accumulation_steps}, gpus={config.world_size})")

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def train(self):
        """Run training loop."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60 + "\n")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}\n")

            self.train_epoch()

            # Final validation
            val_loss = self.evaluate()
            print(f"\nEpoch {epoch + 1} validation loss: {val_loss:.4f}")

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)

        # Save final checkpoint
        self.save_checkpoint("final")

    def train_epoch(self):
        """Train for one epoch."""
        # Set epoch for distributed sampler (ensures proper shuffling)
        if self.config.distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

        self.model.train()

        total_loss = 0
        num_batches = 0

        # Only show progress bar on main process
        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            disable=not self.is_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            images = batch["images"].to(self.config.device)

            # Forward pass with mixed precision
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    labels=labels,
                )

                loss = outputs["loss"] / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients (handle DDP wrapper)
                model_unwrapped = self.model.module if self.config.distributed else self.model
                torch.nn.utils.clip_grad_norm_(model_unwrapped.adapter.parameters(), max_norm=1.0)

                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Log metrics (only on main process)
                if self.is_main_process and self.global_step % self.config.log_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                        "train/lr": lr,
                        "train/epoch": self.current_epoch,
                        "train/step": self.global_step,
                    })

                # Evaluate (only on main process to avoid duplicate validation)
                if self.is_main_process and self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    print(f"\nStep {self.global_step} - Val loss: {val_loss:.4f}")
                    self.model.train()  # Back to training mode

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating", disable=not self.is_main_process):
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            images = batch["images"].to(self.config.device)

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    labels=labels,
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Log to wandb (only on main process)
        if self.is_main_process:
            wandb.log({
                "val/loss": avg_loss,
                "val/step": self.global_step,
            })

        return avg_loss

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return  # Only save on rank 0

        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}_step{self.global_step}.pt")

        # Unwrap DDP if needed
        model_unwrapped = self.model.module if self.config.distributed else self.model

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "adapter_state_dict": model_unwrapped.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
        }

        # Atomic save: write to .tmp then rename
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints (keep last N)
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            Path(self.config.checkpoint_dir).glob("checkpoint_step*.pt"),
            key=lambda x: x.stat().st_mtime,
        )

        # Remove oldest checkpoints if we exceed the limit
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")


def setup_distributed():
    """Initialize distributed training from SLURM environment."""
    # Check if running in SLURM with multiple tasks
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Set device
        torch.cuda.set_device(local_rank)

        return True, local_rank, world_size

    # Check for torchrun/torch.distributed.launch
    elif "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return True, local_rank, world_size

    # Single GPU training
    return False, -1, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Phase 2a Training - Adapter Alignment")

    # Config file or individual args
    parser.add_argument("--config", type=str, help="Path to config YAML file")

    # Override individual settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/phase2a")

    args = parser.parse_args()

    # Setup distributed training if applicable
    distributed, local_rank, world_size = setup_distributed()

    if distributed:
        print(f"Distributed training initialized: rank {local_rank}/{world_size}")

    # Load config
    config = TrainingConfig()
    config.distributed = distributed
    config.local_rank = local_rank
    config.world_size = world_size

    # Override with command line args
    if args.batch_size:
        config.batch_size_per_gpu = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    try:
        # Initialize trainer
        trainer = Trainer(config)

        # Start training
        trainer.train()
    finally:
        # Cleanup distributed training
        cleanup_distributed()


if __name__ == "__main__":
    main()
