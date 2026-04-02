"""
transformer/layers/positional_encoding.py
==========================================
§ 3.5 — Positional Encoding (Fixed Sinusoidal)

"Since our model contains no recurrence and no convolution, in order for the
model to make use of the order of the sequence, we must inject some
information about the relative or absolute position of the tokens in the
sequence.  To this end, we add 'positional encodings' to the input
embeddings at the bottoms of the encoder and decoder stacks."
                                                    — Vaswani et al. (2017)

Formulae:
    PE(pos, 2i)   = sin( pos / 10000^{2i / d_model} )
    PE(pos, 2i+1) = cos( pos / 10000^{2i / d_model} )

The encodings are *fixed* (not learned) and are registered as a non-trainable
buffer so they automatically move with the model to the target device.

§ 3.4 also notes: "In the embedding layers, we multiply those weights by
√d_model."  That scaling is applied in the Encoder/Decoder __init__, not here.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding (§ 3.5).

    Args:
        d_model : Model dimensionality.
        max_len : Maximum sequence length to pre-compute encodings for.
        dropout : Dropout probability applied after adding PE to embeddings.
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # ── Build the (max_len, d_model) encoding table ───────────────────────
        pe: torch.Tensor = torch.zeros(max_len, d_model)

        # position indices: (max_len, 1)
        position: torch.Tensor = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)

        # Compute 10000^{2i/d_model} in log-space for numerical stability:
        #   div_term[i] = exp( 2i · (−log(10000) / d_model) )
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )  # shape: (d_model/2,)

        # PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension → (1, max_len, d_model) for easy broadcasting
        pe = pe.unsqueeze(0)

        # Register as a non-trainable buffer (moves with model.to(device))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.

        Args:
            x : Token embeddings of shape (batch, T, d_model).

        Returns:
            Embeddings + positional encodings, shape (batch, T, d_model).
        """
        # self.pe has shape (1, max_len, d_model); slice to T
        x = x + self.pe[:, : x.size(1), :]   # type: ignore[index]
        return self.dropout(x)

