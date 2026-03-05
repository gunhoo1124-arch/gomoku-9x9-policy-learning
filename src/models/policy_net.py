"""Policy network for Gomoku move prediction."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class PolicyNet(nn.Module):
    """Small CNN policy model mapping `(3, 9, 9)` -> `81` logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 81),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def mask_illegal_logits(logits: torch.Tensor, legal_actions: Sequence[int]) -> torch.Tensor:
    """
    Mask illegal actions for inference.

    Args:
        logits: Tensor of shape `(batch, 81)` or `(81,)`.
        legal_actions: Action indices that are legal in current state.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_back = True
    elif logits.dim() == 2:
        squeeze_back = False
    else:
        raise ValueError(f"Expected logits dim 1 or 2, got shape {tuple(logits.shape)}.")

    if not legal_actions:
        raise ValueError("Cannot mask logits: no legal actions provided.")

    mask = torch.zeros(81, dtype=torch.bool, device=logits.device)
    mask[list(legal_actions)] = True

    masked = logits.clone()
    masked[:, ~mask] = -1e9
    return masked.squeeze(0) if squeeze_back else masked
