"""
src/models/multitask.py
========================
Multitask model: shared backbone + separate crystal system and space group heads.

Architecture
------------
                       +---> CrystalSystemHead -> crystal_logits  [B, 7]
  Input [B,1,L] -> Backbone
                       +---> SpaceGroupHead   -> space_group_logits [B, 10]

Backbone options
----------------
  "cnn1d"        : CNN1D backbone (default, fast, ~684K params)
  "resnet1d"     : ResNet1D backbone (~1.1M params)
  "cnn_attention": CNNAttention backbone (~900K params)

Output dict
-----------
  {
      "crystal_logits":     Tensor [B, num_crystal_classes],
      "space_group_logits": Tensor [B, num_sg_classes],
  }

Usage
-----
    model = MultitaskModel(
        backbone="cnn1d",
        input_length=4500,
        num_crystal_classes=7,
        num_sg_classes=10,
    )
    out = model(x)   # x: [B, 1, 4500]
    crystal_logits     = out["crystal_logits"]      # [B, 7]
    space_group_logits = out["space_group_logits"]  # [B, 10]
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .cnn1d import CNN1D
from .resnet1d import ResNet1D
from .cnn_attention import CNNAttention
from .heads import CrystalSystemHead, SpaceGroupHead


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------
# Each wrapper exposes:
#   .forward_features(x) -> [B, feat_dim]   (backbone output, no classifier)
#   .feat_dim             -> int

class _CNN1DBackbone(nn.Module):
    def __init__(self, input_length: int, **kwargs):
        super().__init__()
        # Build with a dummy num_classes; we discard the head
        self._base = CNN1D(input_length=input_length, num_classes=1, **kwargs)
        # Backbone output = pool -> flatten, before the head Linear
        # CNN1D.head = [Flatten, Linear(flat_dim, fc_dim), BN, ReLU, Dropout, Linear(fc_dim, nc)]
        # We want features after the second-to-last block (BN->ReLU)
        # Easier: reconstruct the feature extractor portion
        fc_dim   = kwargs.get("fc_dim", 256)
        channels = kwargs.get("channels", [64, 128, 256])
        adap     = kwargs.get("adaptive_pool", 1)
        flat_dim = channels[-1] * adap
        self.feat_dim = fc_dim

        # Extract backbone + pool from CNN1D
        self.backbone = self._base.backbone
        self.pool     = self._base.pool
        # Rebuild feature projection (Linear -> BN -> ReLU) without classifier
        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # [B, C_last, T]
        x = self.pool(x)       # [B, C_last, 1]
        return self.feat_proj(x)  # [B, fc_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class _ResNet1DBackbone(nn.Module):
    def __init__(self, input_length: int, **kwargs):
        super().__init__()
        channels = kwargs.get("channels", [64, 128, 256, 256])
        adap     = kwargs.get("adaptive_pool", 1)
        self.feat_dim = channels[-1] * adap
        # Build ResNet1D without the fc layer
        self._base = ResNet1D(
            input_length=input_length,
            num_classes=1,  # placeholder; fc is not used
            **kwargs,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._base.forward_features(x)  # [B, feat_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class _CNNAttentionBackbone(nn.Module):
    def __init__(self, input_length: int, **kwargs):
        super().__init__()
        d_model = kwargs.get("d_model", 128)
        adap    = kwargs.get("adaptive_pool", 1)
        self.feat_dim = d_model * adap
        self._base = CNNAttention(
            input_length=input_length,
            num_classes=1,  # placeholder
            **kwargs,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._base.forward_features(x)  # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


_BACKBONES = {
    "cnn1d":         _CNN1DBackbone,
    "resnet1d":      _ResNet1DBackbone,
    "cnn_attention": _CNNAttentionBackbone,
}


# ---------------------------------------------------------------------------
# MultitaskModel
# ---------------------------------------------------------------------------

class MultitaskModel(nn.Module):
    """
    Shared-backbone multitask model for crystallographic classification.

    Parameters
    ----------
    backbone             : backbone architecture name
                           ("cnn1d" | "resnet1d" | "cnn_attention")
    input_length         : length of 1-D input signal (default 4500)
    num_crystal_classes  : number of crystal system classes (default 7)
    num_sg_classes       : number of space group classes (default 10)
    head_hidden_dim      : hidden dim inside each head (None = single linear)
    head_dropout         : dropout in each head
    backbone_kwargs      : dict of extra kwargs forwarded to the backbone

    Forward returns
    ---------------
    dict with keys:
        "crystal_logits"     : FloatTensor [B, num_crystal_classes]
        "space_group_logits" : FloatTensor [B, num_sg_classes]
    """

    def __init__(
        self,
        backbone: str = "cnn1d",
        input_length: int = 4500,
        num_crystal_classes: int = 7,
        num_sg_classes: int = 10,
        head_hidden_dim: Optional[int] = 128,
        head_dropout: float = 0.3,
        backbone_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        backbone_name = backbone.lower()
        if backbone_name not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from {list(_BACKBONES.keys())}"
            )
        kwargs = backbone_kwargs or {}
        self._backbone_module = _BACKBONES[backbone_name](
            input_length=input_length, **kwargs
        )
        feat_dim = self._backbone_module.feat_dim

        self.crystal_head = CrystalSystemHead(
            in_features=feat_dim,
            num_classes=num_crystal_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        self.sg_head = SpaceGroupHead(
            in_features=feat_dim,
            num_classes=num_sg_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

        self.backbone_name        = backbone_name
        self.num_crystal_classes  = num_crystal_classes
        self.num_sg_classes       = num_sg_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, 1, L] -> dict of logits."""
        feats = self._backbone_module.forward_features(x)  # [B, feat_dim]
        return {
            "crystal_logits":     self.crystal_head(feats),
            "space_group_logits": self.sg_head(feats),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        """Iterable over backbone-only parameters."""
        return self._backbone_module.parameters()

    def head_parameters(self):
        """Iterable over head parameters only."""
        return list(self.crystal_head.parameters()) + list(self.sg_head.parameters())
