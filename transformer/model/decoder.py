"""
transformer/model/decoder.py
==============================
§ 3.1 — Decoder Stack

"The decoder is also composed of a stack of N = 6 identical layers."

The decoder pipeline for a target (shifted-right) sequence:
    1. Token Embedding  ×  √d_model            (§ 3.4 embedding scaling)
    2. Positional Encoding                      (§ 3.5 sinusoidal)
    3. N × DecoderLayer                         (§ 3.1 Post-LN)
       ├── Masked Self-Attention (causal mask)
       ├── Cross-Attention  (K, V from encoder output)
       └── Feed-Forward

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..layers.decoder_layer import DecoderLayer
from ..layers.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    """
    Full Transformer Decoder (§ 3.1).

    Args:
        vocab_size   : Target vocabulary size.
        d_model      : Model dimensionality.
        n_layers     : Number of stacked decoder layers (N; paper: 6).
        n_heads      : Number of attention heads.
        d_ff         : Feed-forward inner dimensionality.
        dropout      : Dropout probability.
        max_seq_len  : Maximum sequence length for the PE table.
        pad_token_id : Padding token ID.
    """

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int,
        n_layers:     int,
        n_heads:      int,
        d_ff:         int,
        dropout:      float = 0.1,
        max_seq_len:  int   = 5000,
        pad_token_id: int   = 0,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model

        # Token embedding table (§ 3.4)
        self.embedding: nn.Embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        # Sinusoidal positional encoding (§ 3.5)
        self.pos_enc: PositionalEncoding = PositionalEncoding(
            d_model, max_seq_len, dropout
        )

        # N identical decoder layers
        self.layers: nn.ModuleList = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        trg:         torch.Tensor,
        encoder_out: torch.Tensor,
        trg_mask:    Optional[torch.Tensor] = None,
        src_mask:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Decode with cross-attention over encoder output.

        Args:
            trg         : (batch, T_trg) integer target token IDs (teacher-forced)
            encoder_out : (batch, T_src, d_model) encoder representations
            trg_mask    : (batch, 1, T_trg, T_trg) causal + padding mask
            src_mask    : (batch, 1, 1, T_src) source padding mask

        Returns:
            x                    : (batch, T_trg, d_model) decoder output
            self_attn_maps       : per-layer self-attention weights
                                   [(batch, n_heads, T_trg, T_trg)] × N
            cross_attn_maps      : per-layer cross-attention weights
                                   [(batch, n_heads, T_trg, T_src)] × N
        """
        # § 3.4: scale embedding by √d_model
        x: torch.Tensor = self.embedding(trg) * math.sqrt(self.d_model)
        # § 3.5: add positional encodings
        x = self.pos_enc(x)

        self_attn_maps:  List[torch.Tensor] = []
        cross_attn_maps: List[torch.Tensor] = []

        for layer in self.layers:
            x, self_w, cross_w = layer(
                x, encoder_out, trg_mask=trg_mask, src_mask=src_mask
            )
            self_attn_maps.append(self_w)
            cross_attn_maps.append(cross_w)

        return x, self_attn_maps, cross_attn_maps

