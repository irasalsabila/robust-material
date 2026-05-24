"""
src/models/mlp.py

MLP baseline for 1-D XRD pattern classification.

Architecture
------------
Input  : [B, 1, L]  →  flatten  →  [B, L]
Layers : Linear(L, 512) → BN → ReLU → Dropout
         Linear(512, 256) → BN → ReLU → Dropout
         Linear(256, 128) → BN → ReLU → Dropout
         Linear(128, num_classes)

Total params (L=4500, 7 classes): ~2.5 M
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Parameters
    ----------
    input_length : length of the 1-D pattern (default 4500)
    num_classes  : number of output classes
    hidden_dims  : list of hidden layer widths
    dropout      : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_length: int = 4500,
        num_classes: int = 7,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers: List[nn.Module] = [nn.Flatten()]  # [B,1,L] -> [B,L]

        in_dim = input_length
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 1, L]  →  logits : [B, num_classes]"""
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
