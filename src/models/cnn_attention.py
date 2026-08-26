from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import CrystalSystemHead

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
    """Learnable positional embeddings for [B, T, d_model]."""

    def __init__(self, d_model: int, max_len: int = 300) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)

        if T > self.pe.num_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds positional embedding length "
                f"{self.pe.num_embeddings}. Increase input_length or max_len."
            )

        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)

# ---------------------------------------------------------------------------
# CNNAttention
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over sequence positions.

    Input : [B, T, d_model]
    Output: [B, d_model]
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)  # [B, T, 1]
        return torch.sum(x * weights, dim=1)           # [B, d_model]


class CNNAttention(nn.Module):
    """
    CNN + Transformer attention model for XRD classification.

    Input : [B, 1, L]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        input_length: int = 4500,
        num_classes: int = 7,
        cnn_channels: List[int] | None = None,
        kernel_size: int = 15,
        pool_size: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dim: int = 256,
        num_attn_layers: int = 1,
        dropout_conv: float = 0.1,
        dropout_head: float = 0.3,
        head_hidden_dim: int | None = 256,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )

        # CNN backbone
        conv_layers: List[nn.Module] = []
        in_ch = 1

        for out_ch in cnn_channels:
            conv_layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout=dropout_conv,
                )
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*conv_layers)

        # Channel projection before attention
        self.proj = nn.Sequential(
            nn.Conv1d(cnn_channels[-1], d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        # After repeated MaxPool1d(pool_size), length is roughly:
        # input_length / pool_size ** num_blocks
        max_seq_len = math.ceil(input_length / (pool_size ** len(cnn_channels))) + 16
        self.pos_enc = LearnablePositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attn_layers,
        )

        # Better than plain average pooling for XRD peaks
        self.attn_pool = AttentionPooling(d_model=d_model)

        # Reuse your project head
        self.head = CrystalSystemHead(
            in_features=d_model,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=dropout_head,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            # Do not re-initialize Linear layers inside Transformer too aggressively.
            # CrystalSystemHead already initializes its own Linear layers.

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return features before classifier.

        x: [B, 1, L]
        returns: [B, d_model]
        """
        x = self.cnn(x)          # [B, C, T]
        x = self.proj(x)         # [B, d_model, T]
        x = x.permute(0, 2, 1)   # [B, T, d_model]

        x = self.pos_enc(x)      # [B, T, d_model]
        x = self.transformer(x)  # [B, T, d_model]

        x = self.attn_pool(x)    # [B, d_model]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)