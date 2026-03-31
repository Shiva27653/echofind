"""
EchoFind — Audio Encoder and SimCLR Model

ResNet-18 backbone adapted for single-channel log-mel spectrograms,
with a projection head for contrastive pre-training.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AudioEncoder(nn.Module):
    """ResNet-18 encoder for single-channel log-mel spectrograms.

    Modifications from standard ResNet-18:
        - conv1 accepts 1-channel input instead of 3-channel RGB
        - fc layer replaced with nn.Identity() to expose the 512-d feature vector

    Args:
        pretrained: Whether to load ImageNet-pretrained weights (default: False).

    Returns:
        Tensor of shape (B, 512) — L2-normalizable embedding.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Log-mel spectrogram tensor of shape (B, 1, 128, T).

        Returns:
            Feature vector of shape (B, 512).
        """
        return self.backbone(x)


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR contrastive learning.

    Maps encoder representations to a lower-dimensional space where
    the NT-Xent loss is applied. Discarded after pre-training.

    Architecture: Linear(512→256) → BN → ReLU → Linear(256→128)

    Args:
        in_dim:     Input dimensionality (default: 512).
        hidden_dim: Hidden layer width (default: 256).
        out_dim:    Output embedding dimensionality (default: 128).
    """

    def __init__(
        self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Encoder output of shape (B, in_dim).

        Returns:
            Projected embedding of shape (B, out_dim).
        """
        return self.net(x)


class SimCLRModel(nn.Module):
    """Full SimCLR model combining encoder and projection head.

    During training both components are active. At inference time only
    the encoder is used; the projection head is discarded.

    Args:
        pretrained: Whether the encoder loads ImageNet weights.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = AudioEncoder(pretrained=pretrained)
        self.projector = ProjectionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder + projection head.

        Args:
            x: Log-mel spectrogram of shape (B, 1, 128, T).

        Returns:
            Projected embedding of shape (B, 128).
        """
        h = self.encoder(x)
        z = self.projector(h)
        return z
