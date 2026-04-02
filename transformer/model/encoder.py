"""
transformer/model/encoder.py
==============================
§ 3.1 — Encoder Stack

"The encoder is composed of a stack of N = 6 identical layers."

The encoder pipeline for a source sequence:
    1. Token Embedding  ×  √d_model            (§ 3.4 embedding scaling)
    2. Positional Encoding                      (§ 3.5 sinusoidal)
    3. N × EncoderLayer  (self-attn + FFN)      (§ 3.1 Post-LN)

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..layers.encoder_layer import EncoderLayer
from ..layers.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    """
    Full Transformer Encoder (§ 3.1).

    Args:
        vocab_size   : Source vocabulary size.
        d_model      : Model dimensionality.
        n_layers     : Number of stacked encoder layers (N; paper: 6).
        n_heads      : Number of attention heads.
        d_ff         : Feed-forward inner dimensionality.
        dropout      : Dropout probability.
        max_seq_len  : Maximum sequence length for the PE table.
        pad_token_id : Padding token ID (used for padding_idx in Embedding).
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

        # N identical encoder layers
        self.layers: nn.ModuleList = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        src:      torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode source tokens.

        Args:
            src      : (batch, T_src) integer token IDs
            src_mask : (batch, 1, 1, T_src) padding mask

        Returns:
            x             : (batch, T_src, d_model) encoded representations
            attn_maps     : List of attention-weight tensors (one per layer),
                            each of shape (batch, n_heads, T_src, T_src)
        """
        # § 3.4: "In the embedding layers, we multiply those weights by √d_model"
        x: torch.Tensor = self.embedding(src) * math.sqrt(self.d_model)
        # § 3.5: add sinusoidal positional encodings + dropout
        x = self.pos_enc(x)

        attn_maps: List[torch.Tensor] = []
        for layer in self.layers:
            x, weights = layer(x, src_mask)
            attn_maps.append(weights)

        return x, attn_maps

