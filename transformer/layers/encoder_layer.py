"""
transformer/layers/encoder_layer.py
=====================================
§ 3.1 — Single Encoder Layer

"The encoder is composed of a stack of N = 6 identical layers.  Each layer
has two sub-layers.  The first is a multi-head self-attention mechanism, and
the second is a simple, position-wise fully connected feed-forward network.
We employ a residual connection around each of the two sub-layers, followed
by layer normalization."                             — Vaswani et al. (2017)

    Sub-layer output = LayerNorm( x + Dropout( Sublayer(x) ) )

──────────────────────────────────────────────────────────────────────────────
NOTE ON LAYERNORM PLACEMENT
──────────────────────────────────────────────────────────────────────────────
This implementation uses **Post-LN** exactly as described in the original
paper: normalisation is applied *after* the residual addition.

    Post-LN (this file):  out = LayerNorm( x + Sublayer(x) )
    Pre-LN  (modern alt): out = x + Sublayer( LayerNorm(x) )

Pre-LN is generally more stable for deep networks and is preferred in modern
practice (e.g. GPT-2), but Post-LN matches Vaswani et al. (2017).
──────────────────────────────────────────────────────────────────────────────

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..attention.multi_head import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward as FeedForward

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer — Post-LN variant (§ 3.1).

    Sub-layers:
        1. Multi-Head Self-Attention
        2. Position-Wise Feed-Forward

    Each wrapped as:  LayerNorm( x + Dropout( Sublayer(x) ) )

    Args:
        d_model  : Model dimensionality.
        n_heads  : Number of attention heads.
        d_ff     : Feed-forward inner dimensionality.
        dropout  : Dropout probability.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Sub-layer 1: Multi-Head Self-Attention (§ 3.2)
        self.self_attn: MultiHeadAttention = MultiHeadAttention(
            d_model, n_heads, dropout
        )
        # Sub-layer 2: Position-Wise Feed-Forward (§ 3.3)
        self.feed_forward: PositionWiseFeedForward = PositionWiseFeedForward(
            d_model, d_ff, dropout
        )

        # Post-LN: LayerNorm applied *after* each sub-layer's residual (§ 3.1)
        self.norm_1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm_2: nn.LayerNorm = nn.LayerNorm(d_model)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through one encoder layer.

        Args:
            x        : (batch, T_src, d_model) — input from previous layer
            src_mask : (batch, 1, 1, T_src) — padding mask
                       (1 at PAD positions, 0 elsewhere)

        Returns:
            x       : (batch, T_src, d_model) — transformed output
            weights : (batch, n_heads, T_src, T_src) — self-attention weights
        """
        # ── Sub-layer 1: Self-Attention ──────────────────────────────────────
        # Q = K = V = x  (self-attention)
        attn_out, weights = self.self_attn(x, x, x, mask=src_mask)
        # Post-LN residual: LayerNorm( x + Dropout(attn_out) )
        x = self.norm_1(x + self.dropout(attn_out))

        # ── Sub-layer 2: Feed-Forward ────────────────────────────────────────
        ff_out: torch.Tensor = self.feed_forward(x)
        # Post-LN residual: LayerNorm( x + Dropout(ff_out) )
        x = self.norm_2(x + self.dropout(ff_out))

        return x, weights

