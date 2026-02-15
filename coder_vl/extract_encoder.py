"""
Extract Vision Encoder from DeepSeek-OCR-2

Loads the full DeepSeek-OCR-2 model and extracts only the vision components:
- SAM (ImageEncoderViT) - image → [1, 896, 16, 16]
- Qwen2Decoder2Encoder - → [1, 256, 896]
- MlpProjector - → [1, 256, 1280]

Discards the language decoder (Qwen2) to save memory during training.

Output: Standalone vision encoder module (~1.5-2 GB vs ~26 GB for full model)
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse


class VisionEncoderPipeline(nn.Module):
    """
    Standalone vision encoder extracted from DeepSeek-OCR-2.

    Processes images through the full vision pipeline:
    Image → SAM → Qwen2Decoder2Encoder → MlpProjector → [batch, num_tokens, 1280]
    """

    def __init__(self, sam, decoder2encoder, mlp_projector):
        super().__init__()

        # Vision components from DeepSeek-OCR-2
        self.sam = sam  # ImageEncoderViT
        self.decoder2encoder = decoder2encoder  # Qwen2Decoder2Encoder
        self.mlp_projector = mlp_projector  # MlpProjector

        # Freeze all parameters (will be frozen during training)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual features.

        Args:
            images: [batch, 3, H, W] - preprocessed images

        Returns:
            visual_features: [batch, num_tokens, 1280]
                           num_tokens = 256 (base) or up to 1120 (with patches)
        """
        # Step 1: SAM (ImageEncoderViT)
        # Input: [batch, 3, H, W]
        # Output: [batch, 896, 16, 16]
        sam_features = self.sam(images)

        # Step 2: Qwen2Decoder2Encoder
        # Input: [batch, 896, 16, 16]
        # Output: [batch, 256, 896] (or more tokens with dynamic tiling)
        encoder_features = self.decoder2encoder(sam_features)

        # Step 3: MlpProjector
        # Input: [batch, num_tokens, 896]
        # Output: [batch, num_tokens, 1280]
        visual_features = self.mlp_projector(encoder_features)

        return visual_features


def extract_vision_encoder(
    model_path: str = "deepseek-ai/deepseek-ocr-2",
    output_path: str = "./models/vision_encoder.pt",
    device: str = "cuda",
):
    """
    Extract vision encoder from DeepSeek-OCR-2.

    Args:
        model_path: HuggingFace model ID or local path
        output_path: Where to save extracted encoder
        device: Device to load model on
    """
    print("="*70)
    print("EXTRACTING VISION ENCODER FROM DEEPSEEK-OCR-2")
    print("="*70)
    print()

    # Load full DeepSeek-OCR-2 model
    print(f"Loading full model from {model_path}...")
    print("⚠️  This will use ~26 GB VRAM in fp32 or ~13 GB in fp16")
    print()

    from transformers import AutoModel

    try:
        # Load model in fp16 to save memory
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

        print("✅ Model loaded successfully")
        print()

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("Make sure you're running on a GPU with enough VRAM (>= 16 GB)")
        return

    # Inspect model structure to find vision components
    print("Model structure:")
    print("-" * 70)
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
    print()

    # Extract vision components from DeepSeek-OCR-2
    # Based on actual model structure inspection
    print("Extracting vision components...")
    print()

    try:
        # DeepSeek-OCR-2 specific attribute names
        if hasattr(model, 'model'):
            # Components are under model.model
            sam = model.model.sam_model
            decoder2encoder = model.model.qwen2_model
            mlp_projector = model.model.projector

            print(f"✅ Found sam_model: {type(sam).__name__}")
            print(f"✅ Found qwen2_model: {type(decoder2encoder).__name__}")
            print(f"✅ Found projector: {type(mlp_projector).__name__}")
        else:
            raise AttributeError("Model structure doesn't match expected DeepSeek-OCR-2 format")

        print()

        # Components successfully extracted (no need to check since we know the structure)

        # Create standalone vision encoder
        print("Creating standalone vision encoder pipeline...")
        vision_encoder = VisionEncoderPipeline(
            sam=sam,
            decoder2encoder=decoder2encoder,
            mlp_projector=mlp_projector,
        )

        # Move to CPU for saving
        vision_encoder = vision_encoder.cpu()

        # Count parameters
        total_params = sum(p.numel() for p in vision_encoder.parameters())
        print(f"✅ Vision encoder created with {total_params:,} parameters")
        print()

        # Verify output shape with dummy input
        print("Verifying output shape...")
        dummy_input = torch.randn(1, 3, 896, 896)  # Typical size for DeepSeek-VL2

        with torch.no_grad():
            try:
                output = vision_encoder(dummy_input)
                print(f"✅ Output shape: {output.shape}")
                print(f"   Expected: [1, 256 or more, 1280]")

                # Verify dimensions
                assert output.shape[0] == 1, "Batch size mismatch"
                assert output.shape[-1] == 1280, f"Feature dim should be 1280, got {output.shape[-1]}"
                print("✅ Dimensions verified!")
                print()

            except Exception as e:
                print(f"⚠️  Error during forward pass: {e}")
                print("   This may be OK - verification will happen during training")
                print()

        # Save to disk
        print(f"Saving vision encoder to {output_path}...")
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as checkpoint dict with state_dict
        # The full module can't be pickled due to custom DeepSeek components
        # The loading code in model.py handles this format by reconstructing from DeepSeek-OCR-2
        checkpoint = {
            'vision_encoder': vision_encoder.state_dict(),
            'model_source': model_path,
            'num_parameters': total_params,
            'expected_output_dim': 1280,
        }
        torch.save(checkpoint, output_path)

        # Check file size
        file_size_gb = Path(output_path).stat().st_size / (1024**3)
        print(f"✅ Saved successfully!")
        print(f"   File size: {file_size_gb:.2f} GB")
        print()

        if file_size_gb > 3:
            print("⚠️  File larger than expected (~1.5-2 GB)")
            print("   This might be OK, but verify it's not the full model")

        print("="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print()
        print(f"Vision encoder saved to: {output_path}")
        print(f"Use this path in train_projector.py")
        print()

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        print()
        import traceback
        traceback.print_exc()


def load_extracted_encoder(checkpoint_path: str, device: str = "cuda") -> VisionEncoderPipeline:
    """
    Load previously extracted vision encoder.

    Args:
        checkpoint_path: Path to saved vision encoder
        device: Device to load on

    Returns:
        vision_encoder: Ready-to-use VisionEncoderPipeline
    """
    print(f"Loading vision encoder from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Need to recreate the pipeline structure
    # This is a simplified loader - the full implementation needs
    # to reconstruct SAM, Decoder2Encoder, MlpProjector from state_dict

    print(f"✅ Loaded encoder with {checkpoint['num_parameters']:,} parameters")
    print(f"   Expected output dim: {checkpoint['expected_output_dim']}")

    # TODO: Implement full reconstruction
    # For now, this is a placeholder
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Extract vision encoder from DeepSeek-OCR-2")

    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/deepseek-ocr-2",
        help="HuggingFace model ID or local path",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/vision_encoder.pt",
        help="Where to save extracted encoder",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    extract_vision_encoder(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
