"""
src/models/resnet1d.py
======================
1-D ResNet baseline for XRD pattern classification.

Architecture
------------
Input  : [B, 1, L]

Stem   : Conv1d(1, C0, kernel=15, stride=2, padding=7) -> BN -> ReLU
         MaxPool1d(kernel=3, stride=2, padding=1)

Stages : configurable number of residual blocks per stage
         each block: Conv -> BN -> ReLU -> Conv -> BN + skip connection
         stride=2 at first block of each stage after stage-0 (downsampling)

Head   : AdaptiveAvgPool1d(1) -> Flatten -> Dropout -> Linear(C_last, num_classes)

Design notes
------------
* Supports variable depth via `layers` list, e.g. [2,2,2,2] (ResNet-8-like)
  or [3,4,6,3] (ResNet-18-like scaled to 1D).
* Bottleneck blocks not implemented (overkill for XRD sequences).
* All channels configurable; defaults give ~1.1M params at L=4500, 7 classes.
* Backbone is exposed separately (self.stem + self.stages) for use in
  MultitaskModel.

Usage
-----
    model = ResNet1D(input_length=4500, num_classes=7)
    logits = model(x)   # x: [B, 1, 4500]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Basic residual block for 1-D sequences.

    two 3x1 (kernel=15) convolutions with batch norm and skip connection.
    Downsampling is applied to both the residual path and skip when stride>1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection: 1x1 conv if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResNet1D
# ---------------------------------------------------------------------------

class ResNet1D(nn.Module):
    """
    Parameters
    ----------
    input_length  : length of 1-D input (default 4500)
    num_classes   : output classes
    layers        : list of residual blocks per stage, e.g. [2, 2, 2, 2]
    channels      : list of channel widths per stage, e.g. [64, 128, 256, 512]
                    Must be the same length as `layers`.
    kernel_size   : conv kernel size inside residual blocks (default 15)
    stem_channels : channels in the stem conv (default 64)
    dropout_res   : dropout within residual blocks
    dropout_head  : dropout before classification linear
    adaptive_pool : output size of AdaptiveAvgPool1d (default 1)
    """

    def __init__(
        self,
        input_length: int = 4500,
        num_classes: int = 7,
        layers: List[int] = None,
        channels: List[int] = None,
        kernel_size: int = 15,
        stem_channels: int = 64,
        dropout_res: float = 0.1,
        dropout_head: float = 0.3,
        adaptive_pool: int = 1,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]          # ~ResNet-18 depth scaled to 1D
        if channels is None:
            channels = [64, 128, 256, 256]  # keep last stage width moderate
        assert len(layers) == len(channels), (
            "layers and channels must have the same length"
        )

        # ── Stem ─────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual stages ───────────────────────────────────────────────
        stages: List[nn.Module] = []
        in_ch = stem_channels
        for stage_idx, (n_blocks, out_ch) in enumerate(zip(layers, channels)):
            stage_blocks: List[nn.Module] = []
            for block_idx in range(n_blocks):
                # Downsample (stride=2) only at first block of non-first stage
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stage_blocks.append(
                    ResBlock1D(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout_res,
                    )
                )
                in_ch = out_ch
            stages.append(nn.Sequential(*stage_blocks))
        self.stages = nn.Sequential(*stages)

        # ── Head ──────────────────────────────────────────────────────────
        self.pool    = nn.AdaptiveAvgPool1d(adaptive_pool)
        flat_dim     = channels[-1] * adaptive_pool
        self.dropout = nn.Dropout(p=dropout_head)
        self.fc      = nn.Linear(flat_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flat backbone features [B, C_last] without the classifier."""
        x = self.stem(x)     # [B, stem_ch, L/4]
        x = self.stages(x)   # [B, C_last, L']
        x = self.pool(x)     # [B, C_last, 1]
        return x.flatten(1)  # [B, C_last]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, L] -> logits: [B, num_classes]"""
        feats = self.forward_features(x)
        return self.fc(self.dropout(feats))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
