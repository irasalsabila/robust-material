"""
src/models/cnn1d.py

Small 1-D CNN baseline for XRD pattern classification.

Architecture
------------
Input : [B, 1, L]

Conv block x3:
  Conv1d(C_in, C_out, kernel=15, padding=7) → BN → ReLU → MaxPool(4)

After 3 pools: L → L/64  (4500 → 70 → 17 → 4)

Head:
  AdaptiveAvgPool1d(1) → Flatten → Linear(C3, fc_dim) → BN → ReLU
  → Dropout → Linear(fc_dim, num_classes)

Total params (~L=4500, 7 classes): ~0.2 M
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        pool_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    """
    Parameters
    ----------
    input_length  : length of the 1-D pattern (default 4500)
    num_classes   : number of output classes
    channels      : list of channel widths for each conv block
    kernel_size   : conv kernel size (same for all blocks)
    pool_size     : max-pool stride (same for all blocks)
    fc_dim        : width of the fully-connected hidden layer
    dropout_conv  : dropout after each conv block
    dropout_fc    : dropout before the final linear layer
    adaptive_pool : output size of AdaptiveAvgPool1d before FC
    """

    def __init__(
        self,
        input_length: int = 4500,
        num_classes: int = 7,
        channels: List[int] = None,
        kernel_size: int = 15,
        pool_size: int = 4,
        fc_dim: int = 256,
        dropout_conv: float = 0.1,
        dropout_fc: float = 0.3,
        adaptive_pool: int = 1,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        # ── Convolutional backbone ────────────────────────────────────────
        conv_layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            conv_layers.append(
                ConvBlock(in_ch, out_ch,
                          kernel_size=kernel_size,
                          pool_size=pool_size,
                          dropout=dropout_conv)
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*conv_layers)

        # ── Global pooling + head ─────────────────────────────────────────
        # AdaptiveAvgPool1d(1) = global average pooling; works on all devices
        self.pool = nn.AdaptiveAvgPool1d(adaptive_pool)
        flat_dim  = channels[-1] * adaptive_pool

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(fc_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 1, L]  →  logits : [B, num_classes]"""
        x = self.backbone(x)   # [B, C_last, L']
        x = self.pool(x)       # [B, C_last, adaptive_pool]
        return self.head(x)    # [B, num_classes]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
