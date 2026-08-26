from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .heads import CrystalSystemHead


class MLP(nn.Module):
    def __init__(
        self,
        input_length: int = 4500,
        num_classes: int = 7,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.3,
        head_hidden_dim: int | None = 128,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        layers: List[nn.Module] = [
            nn.Flatten(),              # [B, 1, L] -> [B, L]
            nn.LayerNorm(input_length),
        ]

        in_dim = input_length

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                ]
            )
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        self.head = CrystalSystemHead(
            in_features=in_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector before classification head."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, L] -> logits: [B, num_classes]"""
        feats = self.forward_features(x)
        return self.head(feats)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)