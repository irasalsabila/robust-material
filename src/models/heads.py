"""
src/models/heads.py
====================
Reusable classification heads for crystallographic tasks.

Provides:
  CrystalSystemHead  — 7-class crystal system classifier
  SpaceGroupHead     — N-class space group classifier (default top-10)

Both heads share the same interface:
  __init__(in_features, num_classes, dropout)
  forward(x) -> logits  [B, num_classes]

Designed to be attached to any backbone that produces
a flat feature vector of shape [B, in_features].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrystalSystemHead(nn.Module):
    """
    Classification head for crystal system prediction (7 classes).

    Parameters
    ----------
    in_features  : dimension of the input feature vector
    num_classes  : number of crystal system classes (default 7)
    hidden_dim   : optional hidden projection; None = single linear layer
    dropout      : dropout probability before the final linear
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dim: int | None = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features] -> logits: [B, num_classes]"""
        return self.head(x)


class SpaceGroupHead(nn.Module):
    """
    Classification head for space group prediction.

    Parameters
    ----------
    in_features  : dimension of the input feature vector
    num_classes  : number of space group classes (default 10 for top-10)
    hidden_dim   : optional hidden projection; None = single linear layer
    dropout      : dropout probability before the final linear
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 10,
        hidden_dim: int | None = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features] -> logits: [B, num_classes]"""
        return self.head(x)
