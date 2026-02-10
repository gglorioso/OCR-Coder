"""
Projection Adapter for DeepSeek-Coder-VL

Maps visual features from SigLIP vision encoder (1280D) to
DeepSeek-Coder-V2 embedding space (2048D).

Architecture: 2-layer MLP with GELU activation (13.6M parameters)
- Layer 1: 1280 → 4096 (5.2M params)
- Layer 2: 4096 → 2048 (8.4M params)
"""

import torch
import torch.nn as nn


class ProjectionAdapter(nn.Module):
    """
    MLP-based projection adapter following LLaVA-1.5 design.

    Args:
        vision_dim: Output dimension of vision encoder (default: 1280 for SigLIP)
        hidden_dim: Intermediate hidden dimension (default: 4096, 2x max of in/out)
        coder_dim: Input dimension of coder model (default: 2048 for DeepSeek-Coder-V2-Lite)
    """

    def __init__(self, vision_dim: int = 1280, hidden_dim: int = 4096, coder_dim: int = 2048):
        super().__init__()

        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.coder_dim = coder_dim

        # 2-layer MLP: vision_dim → hidden_dim → coder_dim
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),   # 5.2M params
            nn.GELU(),
            nn.Linear(hidden_dim, coder_dim),    # 8.4M params
        )

        # Initialize weights (standard PyTorch initialization is fine, but explicit for clarity)
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with normal distribution."""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Project visual features to coder embedding space.

        Args:
            visual_features: Tensor of shape [batch_size, num_visual_tokens, vision_dim]
                           Example: [4, 1120, 1280] for batch of 4 images

        Returns:
            projected_features: Tensor of shape [batch_size, num_visual_tokens, coder_dim]
                              Example: [4, 1120, 2048]
        """
        # Input validation
        assert visual_features.dim() == 3, \
            f"Expected 3D input [batch, tokens, dim], got {visual_features.dim()}D"
        assert visual_features.size(-1) == self.vision_dim, \
            f"Expected vision_dim={self.vision_dim}, got {visual_features.size(-1)}"

        # Project through MLP
        projected = self.projector(visual_features)

        # Output validation
        assert projected.size(-1) == self.coder_dim, \
            f"Output dim mismatch: expected {self.coder_dim}, got {projected.size(-1)}"

        return projected

    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_projector():
    """Simple test to verify adapter shape transformation."""
    print("Testing ProjectionAdapter...")

    # Create adapter
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=2048)

    # Print parameter count
    print(f"Total parameters: {adapter.num_parameters():,}")
    print(f"Trainable parameters: {adapter.num_trainable_parameters():,}")

    # Test with dummy input (batch=2, tokens=1120, dim=1280)
    batch_size = 2
    num_tokens = 1120  # Max visual tokens from SigLIP

    dummy_input = torch.randn(batch_size, num_tokens, 1280)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    output = adapter(dummy_input)
    print(f"Output shape: {output.shape}")

    # Verify dimensions
    assert output.shape == (batch_size, num_tokens, 2048), "Output shape mismatch!"
    print("\n✅ Adapter test passed!")

    return adapter


if __name__ == "__main__":
    test_projector()
