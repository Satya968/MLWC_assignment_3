"""The two convolutional networks that make up ChannelNet.

SRCNN (super-resolution).  Three convolutions with no downsampling, so the
72x14 image keeps its size throughout ('same' padding).  It upgrades the
coarse interpolated pilot image into a sharper estimate:

    9x9 conv, 64 maps + ReLU   (patch extraction)
    1x1 conv, 32 maps + ReLU   (non-linear mapping)
    5x5 conv,  1 map           (reconstruction)

DnCNN (denoising).  A deep residual stack that predicts the *noise* left in the
SRCNN output and subtracts it.  One conv+ReLU, a tower of conv+BN+ReLU blocks,
and a final conv back to one channel; the forward pass returns input - residual.
"""
from __future__ import annotations

import torch
from torch import nn


class SRCNN(nn.Module):
    """Dong et al. super-resolution network, single-channel in/out."""

    def __init__(self, feat1: int = 64, feat2: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, feat1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat1, feat2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat2, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DnCNN(nn.Module):
    """Zhang et al. residual denoiser; `depth` total conv layers."""

    def __init__(self, depth: int = 20, width: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(1, width, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(width, 1, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.body(x)      # residual learning: subtract noise estimate


def n_params(module: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
