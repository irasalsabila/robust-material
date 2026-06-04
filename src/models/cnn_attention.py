"""
src/models/cnn_attention.py
============================
Lightweight CNN + multi-head self-attention model for XRD classification.

Architecture
------------
Input  : [B, 1, L]

CNN Feature Extractor:
  3 ConvBlocks (Conv1d -> BN -> ReLU -> Dropout -> MaxPool)
  Channels: 64 -> 128 -> 256 (configurable)

Projection:
  Conv1d(256, d_model, kernel=1)  -- reduce channel dim before attention

Multi-Head Self-Attention:
  Standard scaled dot-product attention over the sequence dimension
  num_heads=4, d_model=128 (configurable)
  Positional encoding: simple learnable 1D embeddings

Pooling + Head:
  AdaptiveAvgPool1d(1) -> Flatten -> Dropout -> Linear(d_model, num_classes)

Design rationale
----------------
Attention allows the model to attend to characteristic peak regions in
XRD patterns (d-spacing peaks) that may be spatially distant. The CNN
front-end provides local feature extraction efficiently before attention.

Total params (~L=4500, d_model=128, 7 classes): ~0.9M

Usage
-----
    model = CNNAttention(input_length=4500, num_classes=7)
    logits = model(x)  # x: [B, 1, 4500]
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv block (reused from CNN1D design)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> Dropout -> MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        pool_size: int = 4,
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


# ---------------------------------------------------------------------------
# Positional encoding (learnable)
# ---------------------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings for sequences up to max_len tokens."""

    def __init__(self, d_model: int, max_len: int = 300) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]"""
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


# ---------------------------------------------------------------------------
# CNNAttention
# ---------------------------------------------------------------------------

class CNNAttention(nn.Module):
    """
    Parameters
    ----------
    input_length   : length of 1-D input (default 4500)
    num_classes    : output classes
    cnn_channels   : channel widths for each conv block
    kernel_size    : conv kernel size
    pool_size      : max-pool stride per block
    d_model        : projection dimension fed into attention
    num_heads      : number of attention heads
    attn_dropout   : dropout inside attention
    ff_dim         : feed-forward expansion inside transformer layer
    num_attn_layers: number of stacked transformer encoder layers (default 1)
    dropout_conv   : dropout inside conv blocks
    dropout_head   : dropout before output linear
    adaptive_pool  : output size of AdaptiveAvgPool1d
    """

    def __init__(
        self,
        input_length: int = 4500,
        num_classes: int = 7,
        cnn_channels: List[int] = None,
        kernel_size: int = 15,
        pool_size: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dim: int = 256,
        num_attn_layers: int = 1,
        dropout_conv: float = 0.1,
        dropout_head: float = 0.3,
        adaptive_pool: int = 1,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        # ── CNN backbone ─────────────────────────────────────────────────
        conv_layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in cnn_channels:
            conv_layers.append(
                ConvBlock(in_ch, out_ch,
                          kernel_size=kernel_size,
                          pool_size=pool_size,
                          dropout=dropout_conv)
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # ── Channel projection: CNN output -> d_model ─────────────────────
        # Use 1x1 conv to project channel dim while keeping sequence dim
        self.proj = nn.Sequential(
            nn.Conv1d(cnn_channels[-1], d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        # ── Positional encoding ───────────────────────────────────────────
        # After 3 max-pools of size 4: L -> L/64  (4500 -> ~70)
        # Add buffer for variable pool outputs
        max_seq_len = math.ceil(input_length / (pool_size ** len(cnn_channels))) + 16
        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=max_seq_len)

        # ── Transformer encoder layers ────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model       = d_model,
            nhead         = num_heads,
            dim_feedforward = ff_dim,
            dropout       = attn_dropout,
            batch_first   = True,  # [B, T, d_model]
            norm_first    = True,  # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_attn_layers
        )

        # ── Pooling + head ────────────────────────────────────────────────
        self.pool    = nn.AdaptiveAvgPool1d(adaptive_pool)
        flat_dim     = d_model * adaptive_pool
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
        """Return flat features [B, d_model] without the classifier head."""
        x = self.cnn(x)          # [B, C_last, T]
        x = self.proj(x)         # [B, d_model, T]
        x = x.permute(0, 2, 1)  # [B, T, d_model]  -- batch_first
        x = self.pos_enc(x)      # [B, T, d_model]
        x = self.transformer(x)  # [B, T, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, T]
        x = self.pool(x)         # [B, d_model, 1]
        return x.flatten(1)      # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, L] -> logits: [B, num_classes]"""
        feats = self.forward_features(x)
        return self.fc(self.dropout(feats))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
