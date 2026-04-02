"""
transformer/layers/feed_forward.py
====================================
§ 3.3 — Position-wise Feed-Forward Networks

"In addition to attention sub-layers, each of the layers in our encoder
and decoder contains a fully connected feed-forward network, which is
applied to each position separately and identically.  This consists of two
linear transformations with a ReLU activation in between."
                                                    — Vaswani et al. (2017)

    FFN(x) = max(0, x W₁ + b₁) W₂ + b₂

The inner dimension d_ff = 2048 in the original paper (4 × d_model = 4 × 512).
The same W₁, W₂ are shared across positions within a single layer, but
different layers have different weight matrices.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (§ 3.3).

    Two linear transformations with a ReLU activation and dropout.
    Applied identically at every sequence position.

    Args:
        d_model : Input/output dimensionality (e.g. 512).
        d_ff    : Inner (hidden) dimensionality (e.g. 2048 = 4 × d_model).
        dropout : Dropout rate applied after the first linear+ReLU (§ 5.4).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # W₁ ∈ ℝ^{d_model × d_ff}
        self.linear_1: nn.Linear = nn.Linear(d_model, d_ff)
        # W₂ ∈ ℝ^{d_ff × d_model}
        self.linear_2: nn.Linear = nn.Linear(d_ff, d_model)
        self.dropout:  nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, T, d_model)

        Returns:
            Tensor of shape (batch, T, d_model)
        """
        # FFN(x) = max(0, x W₁ + b₁) W₂ + b₂
        x = self.linear_1(x)    # (batch, T, d_ff)
        x = F.relu(x)           # ReLU activation
        x = self.dropout(x)     # dropout after first activation (§ 5.4)
        x = self.linear_2(x)    # (batch, T, d_model)
        return x

