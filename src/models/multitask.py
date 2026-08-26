from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .mlp import MLP
from .cnn1d import CNN1D
from .resnet1d import ResNet1D
from .cnn_attention import CNNAttention
from .heads import CrystalSystemHead, SpaceGroupHead


_BACKBONES = {
    "mlp": MLP,
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "cnn_attention": CNNAttention,
}


class MultitaskModel(nn.Module):
    """
    Shared-backbone multitask model.

    Predicts:
      1. crystal system
      2. space group

    Input:
      x: [B, 1, L]

    Output:
      {
          "crystal_logits": [B, num_crystal_classes],
          "space_group_logits": [B, num_sg_classes],
      }
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
                f"Choose from {list(_BACKBONES.keys())}."
            )

        kwargs = backbone_kwargs or {}

        # Build backbone with dummy num_classes.
        # We only use forward_features(), not the backbone classifier.
        self.backbone = _BACKBONES[backbone_name](
            input_length=input_length,
            num_classes=1,
            **kwargs,
        )

        if not hasattr(self.backbone, "forward_features"):
            raise AttributeError(
                f"{backbone_name} must implement forward_features(x)."
            )

        self.feat_dim = self._infer_feat_dim(input_length)

        self.crystal_head = CrystalSystemHead(
            in_features=self.feat_dim,
            num_classes=num_crystal_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

        self.sg_head = SpaceGroupHead(
            in_features=self.feat_dim,
            num_classes=num_sg_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

        self.backbone_name = backbone_name
        self.num_crystal_classes = num_crystal_classes
        self.num_sg_classes = num_sg_classes

    @torch.no_grad()
    def _infer_feat_dim(self, input_length: int) -> int:
        """Infer backbone feature dimension automatically."""
        was_training = self.backbone.training
        self.backbone.eval()

        dummy = torch.zeros(2, 1, input_length)
        feats = self.backbone.forward_features(dummy)

        if was_training:
            self.backbone.train()

        if feats.ndim != 2:
            raise ValueError(
                f"Expected forward_features() to return [B, feat_dim], "
                f"got shape {tuple(feats.shape)}."
            )

        return feats.size(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.forward_features(x)

        return {
            "crystal_logits": self.crystal_head(feats),
            "space_group_logits": self.sg_head(feats),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return list(self.crystal_head.parameters()) + list(self.sg_head.parameters())