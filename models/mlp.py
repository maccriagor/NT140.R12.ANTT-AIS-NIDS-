"""
MLP baseline: 1500 -> 512 -> 256 -> n_classes, ReLU between layers.

Contract (skeleton.py): input float (batch, 1500) in [0, 1]; output logits (batch, n_classes);
the last layer is a single nn.Linear stored as `self.head` so the incremental update can extend it.
Everything except `head` lives in `self.backbone`, so its state_dict keys start with "backbone." and
the head keys with "head." (pipeline.restore_backbone relies on that prefix).
"""
import torch
import torch.nn as nn

from models.registry import N_BYTES, register


@register("mlp")
class MLP(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(N_BYTES, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
