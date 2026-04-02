"""
transformer/layers/decoder_layer.py
=====================================
§ 3.1 — Single Decoder Layer

"The decoder is also composed of a stack of N = 6 identical layers.  In
addition to the two sub-layers in each encoder layer, the decoder inserts a
third sub-layer, which performs multi-head attention over the output of the
encoder stack.  … We also modify the self-attention sub-layer in the decoder
stack to prevent positions from attending to subsequent positions."
                                                    — Vaswani et al. (2017)

Sub-layers:
    1. **Masked** Multi-Head Self-Attention — causal mask prevents peeking ahead
    2. Multi-Head Cross-Attention — Q from decoder, K & V from encoder output
    3. Position-Wise Feed-Forward

Each sub-layer uses **Post-LN** (original paper):
    out = LayerNorm( x + Dropout( Sublayer(x) ) )

See encoder_layer.py for discussion of Post-LN vs Pre-LN.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..attention.multi_head import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer — Post-LN variant (§ 3.1).

    Sub-layers:
        1. Masked Multi-Head Self-Attention  (causal mask)
        2. Multi-Head Cross-Attention         (encoder output as K, V)
        3. Position-Wise Feed-Forward

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

        # Sub-layer 1: Masked Self-Attention (causal)
        self.self_attn:  MultiHeadAttention = MultiHeadAttention(
            d_model, n_heads, dropout
        )
        # Sub-layer 2: Cross-Attention (encoder output → K, V)
        self.cross_attn: MultiHeadAttention = MultiHeadAttention(
            d_model, n_heads, dropout
        )
        # Sub-layer 3: Position-Wise Feed-Forward
        self.feed_forward: PositionWiseFeedForward = PositionWiseFeedForward(
            d_model, d_ff, dropout
        )

        # Post-LN normalisation after each sub-layer
        self.norm_1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm_2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm_3: nn.LayerNorm = nn.LayerNorm(d_model)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:             torch.Tensor,
        encoder_out:   torch.Tensor,
        trg_mask:      Optional[torch.Tensor] = None,
        src_mask:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through one decoder layer.

        Args:
            x            : (batch, T_trg, d_model) — decoder input (previous layer)
            encoder_out  : (batch, T_src, d_model) — encoder output
            trg_mask     : (batch, 1, T_trg, T_trg) — causal + padding mask
            src_mask     : (batch, 1, 1, T_src) — source padding mask

        Returns:
            x             : (batch, T_trg, d_model) — transformed output
            self_weights  : (batch, n_heads, T_trg, T_trg) — self-attn weights
            cross_weights : (batch, n_heads, T_trg, T_src) — cross-attn weights
        """
        # ── Sub-layer 1: Masked Self-Attention ───────────────────────────────
        self_attn_out, self_weights = self.self_attn(
            x, x, x, mask=trg_mask
        )
        # Post-LN residual
        x = self.norm_1(x + self.dropout(self_attn_out))

        # ── Sub-layer 2: Cross-Attention ─────────────────────────────────────
        # Q comes from decoder (x), K and V come from encoder output
        cross_attn_out, cross_weights = self.cross_attn(
            x, encoder_out, encoder_out, mask=src_mask
        )
        # Post-LN residual
        x = self.norm_2(x + self.dropout(cross_attn_out))

        # ── Sub-layer 3: Feed-Forward ────────────────────────────────────────
        ff_out: torch.Tensor = self.feed_forward(x)
        # Post-LN residual
        x = self.norm_3(x + self.dropout(ff_out))

        return x, self_weights, cross_weights

