"""
DeepSeek-Coder-VL Model

Combines SigLIP vision encoder + projection adapter + DeepSeek-Coder-V2-Lite
for vision-enabled code reasoning.

Token integration follows LLaVA placeholder-replacement pattern:
1. Tokenize text with <image> placeholder
2. Encode image through vision encoder + adapter
3. Replace <image> token with projected visual tokens
4. Forward through coder transformer
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

from projector import ProjectionAdapter


class VisionEncoderPipeline(nn.Module):
    """
    Standalone vision encoder extracted from DeepSeek-OCR-2.
    This is a simple wrapper that will be loaded from the extracted checkpoint.
    """

    def __init__(self, sam, decoder2encoder, mlp_projector):
        super().__init__()
        self.sam = sam
        self.decoder2encoder = decoder2encoder
        self.mlp_projector = mlp_projector

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual features."""
        sam_features = self.sam(images)
        encoder_features = self.decoder2encoder(sam_features)
        visual_features = self.mlp_projector(encoder_features)
        return visual_features


class CoderVLModel(nn.Module):
    """
    Vision-enabled DeepSeek-Coder model.

    Architecture:
        Image → VisionEncoder (frozen) → Adapter (trainable) → CoderModel (frozen/LoRA)
    """

    def __init__(
        self,
        vision_encoder_path: str,
        coder_model_path: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        adapter_hidden_dim: int = 4096,
        freeze_vision: bool = True,
        freeze_coder: bool = True,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        print("Initializing CoderVL model...")

        # Load vision encoder (will be extracted from DeepSeek-OCR-2)
        print(f"Loading vision encoder from {vision_encoder_path}...")
        self.vision_encoder = self._load_vision_encoder(vision_encoder_path)
        if freeze_vision:
            self._freeze_model(self.vision_encoder)
            print("  Vision encoder frozen ✓")

        # Load coder model (with optional 8-bit quantization to save VRAM)
        print(f"Loading coder model from {coder_model_path}...")
        if load_in_8bit:
            print("  Using 8-bit quantization (reduces VRAM: 32GB → ~8GB)")

        # For distributed training with 8-bit, we need to specify the device
        # device_map="auto" spreads across all GPUs, which breaks DDP
        if load_in_8bit and device is not None:
            device_map = {"": device}
        elif load_in_8bit:
            device_map = "auto"
        else:
            device_map = None

        self.coder_model = AutoModelForCausalLM.from_pretrained(
            coder_model_path,
            torch_dtype=torch.bfloat16 if not load_in_8bit else None,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            coder_model_path,
            trust_remote_code=True,
        )

        # Add special tokens
        self._add_special_tokens()

        if freeze_coder:
            self._freeze_model(self.coder_model)
            print("  Coder model frozen ✓")

        # Get embedding dimensions
        vision_dim = 1280  # SigLIP output (verified in Phase 1.5)
        coder_dim = self.coder_model.config.hidden_size  # Should be 2048

        # Create projection adapter
        print(f"Creating projection adapter ({vision_dim}D → {coder_dim}D)...")
        self.adapter = ProjectionAdapter(
            vision_dim=vision_dim,
            hidden_dim=adapter_hidden_dim,
            coder_dim=coder_dim,
        )
        print(f"  Adapter initialized with {self.adapter.num_parameters():,} parameters ✓")

        # Cache special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids("<img_start>")
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids("<img_end>")

    def _load_vision_encoder(self, path: str) -> nn.Module:
        """
        Load extracted vision encoder from DeepSeek-OCR-2.

        Expected: Full VisionEncoderPipeline module or checkpoint dict
        Output: [batch, 256, 1280] per image (base view, no tiling)
                Up to [batch, 256*7, 1280] with max 6 patches
        """
        loaded = torch.load(path, map_location="cpu")

        # Handle both checkpoint dict and full module formats
        if isinstance(loaded, nn.Module):
            # Full module format (if successfully pickled)
            return loaded
        elif isinstance(loaded, dict) and 'vision_encoder' in loaded:
            # Checkpoint dict with state_dict (standard format)
            # Reconstructs architecture from DeepSeek-OCR-2 (cached by HuggingFace after first download)
            print("  Loading from checkpoint dict format...")
            print("  (Will download DeepSeek-OCR-2 on first run - cached thereafter)")

            from transformers import AutoModel

            # Load full model to get architecture
            model = AutoModel.from_pretrained(
                loaded.get('model_source', 'deepseek-ai/deepseek-ocr-2'),
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            # Extract vision components
            sam = model.model.sam_model
            decoder2encoder = model.model.qwen2_model
            mlp_projector = model.model.projector

            # Create pipeline
            vision_encoder = VisionEncoderPipeline(sam, decoder2encoder, mlp_projector)

            # Load saved weights
            vision_encoder.load_state_dict(loaded['vision_encoder'])

            return vision_encoder
        else:
            raise ValueError(f"Unexpected format: {type(loaded)}")

    def _add_special_tokens(self):
        """Add vision-specific special tokens to tokenizer."""
        special_tokens = {
            "additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"  Added {num_added} special tokens")

        # Resize token embeddings to accommodate new tokens
        self.coder_model.resize_token_embeddings(len(self.tokenizer))
        print(f"  Resized embeddings to {len(self.tokenizer)} tokens")

    def _freeze_model(self, model: nn.Module):
        """Freeze all parameters in a model."""
        for param in model.parameters():
            param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through vision encoder + adapter.

        Args:
            images: Tensor of shape [batch, 3, H, W] (preprocessed images)

        Returns:
            projected_features: [batch, num_visual_tokens, coder_dim]
        """
        with torch.no_grad():  # Vision encoder is frozen
            visual_features = self.vision_encoder(images)
            # Expected shape: [batch, num_tokens, 1280]
            # num_tokens = 256 (base) or up to 256*7 (with patches)

        # Project to coder embedding space (adapter is trainable)
        projected = self.adapter(visual_features)
        # Output shape: [batch, num_tokens, coder_dim]

        return projected

    def prepare_inputs_with_image(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replace <image> placeholder tokens with projected visual features.

        Args:
            input_ids: [batch, seq_len] - tokenized text with <image> placeholder
            visual_features: [batch, num_visual_tokens, coder_dim] - projected image features

        Returns:
            combined_embeddings: [batch, new_seq_len, coder_dim]
            attention_mask: [batch, new_seq_len]
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Get text embeddings from coder model
        text_embeddings = self.coder_model.get_input_embeddings()(input_ids)
        # Shape: [batch, seq_len, coder_dim]

        # Find <image> token positions
        image_token_mask = input_ids == self.image_token_id
        # Shape: [batch, seq_len]

        # Build combined embeddings for each sample in batch
        combined_embeddings_list = []
        attention_masks_list = []

        for i in range(batch_size):
            # Find image token position in this sample
            image_positions = torch.where(image_token_mask[i])[0]

            if len(image_positions) == 0:
                # No image in this sample (text-only)
                combined_embeddings_list.append(text_embeddings[i])
                attention_masks_list.append(torch.ones(seq_len, device=input_ids.device))
            else:
                # Assume single image per sample for Phase 2a
                image_pos = image_positions[0].item()

                # Split text embeddings at image position
                before_image = text_embeddings[i, :image_pos]      # [image_pos, coder_dim]
                after_image = text_embeddings[i, image_pos+1:]     # [seq_len - image_pos - 1, coder_dim]

                # Concatenate: before + visual + after
                combined = torch.cat([
                    before_image,
                    visual_features[i],  # Insert all visual tokens
                    after_image,
                ], dim=0)

                combined_embeddings_list.append(combined)

                # Create attention mask (all 1s)
                new_len = combined.size(0)
                attention_masks_list.append(torch.ones(new_len, device=input_ids.device))

        # Pad sequences to same length
        max_len = max(emb.size(0) for emb in combined_embeddings_list)

        padded_embeddings = []
        padded_masks = []

        for emb, mask in zip(combined_embeddings_list, attention_masks_list):
            pad_len = max_len - emb.size(0)
            if pad_len > 0:
                # Pad embeddings with zeros
                emb = torch.cat([
                    emb,
                    torch.zeros(pad_len, emb.size(1), device=emb.device, dtype=emb.dtype)
                ], dim=0)
                # Pad attention mask with zeros
                mask = torch.cat([
                    mask,
                    torch.zeros(pad_len, device=mask.device, dtype=mask.dtype)
                ], dim=0)

            padded_embeddings.append(emb)
            padded_masks.append(mask)

        # Stack into batch tensors
        combined_embeddings = torch.stack(padded_embeddings, dim=0)
        attention_mask = torch.stack(padded_masks, dim=0)

        return combined_embeddings, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional image input.

        Args:
            input_ids: [batch, seq_len] - tokenized text (may contain <image> tokens)
            images: [batch, 3, H, W] - preprocessed images (None for text-only)
            labels: [batch, seq_len] - target tokens for loss computation
                   -100 for positions to ignore (user prompt, image tokens)

        Returns:
            dict with keys:
                - loss: scalar tensor (if labels provided)
                - logits: [batch, seq_len, vocab_size]
        """
        if images is not None:
            # Encode images
            visual_features = self.encode_image(images)

            # Replace <image> tokens with visual features
            inputs_embeds, attention_mask = self.prepare_inputs_with_image(
                input_ids, visual_features
            )

            # Forward through coder model with embeddings directly
            outputs = self.coder_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels if labels is not None else None,
                **kwargs,
            )
        else:
            # Text-only forward pass (for text replay in Phase 2b)
            outputs = self.coder_model(
                input_ids=input_ids,
                labels=labels,
                **kwargs,
            )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text from image + prompt.

        Args:
            input_ids: [batch, seq_len] - tokenized prompt with <image> token
            images: [batch, 3, H, W] - preprocessed images
            max_new_tokens: maximum tokens to generate

        Returns:
            generated_ids: [batch, seq_len + new_tokens]
        """
        if images is not None:
            # Encode images
            visual_features = self.encode_image(images)

            # Replace <image> tokens
            inputs_embeds, attention_mask = self.prepare_inputs_with_image(
                input_ids, visual_features
            )

            # Generate from embeddings
            outputs = self.coder_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        else:
            # Text-only generation
            outputs = self.coder_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        return outputs

    def num_trainable_parameters(self) -> int:
        """Count trainable parameters across all components."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("CoderVL model module loaded successfully.")
    print("Use train_projector.py to train the model.")
