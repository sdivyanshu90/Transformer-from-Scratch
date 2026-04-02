"""
transformer/model/transformer.py
==================================
§ 3 — Full Encoder-Decoder Transformer

Assembles the complete architecture described in the paper:

    Encoder: N × EncoderLayer(MultiHead-SelfAttn → FFN)   [Post-LN]
    Decoder: N × DecoderLayer(Masked-SelfAttn → CrossAttn → FFN)  [Post-LN]
    Output:  Linear projection  →  log-softmax (implicit in loss)

Also provides:
    make_src_mask()  — padding mask for encoder input
    make_trg_mask()  — causal (autoregressive) + padding mask for decoder
    greedy_decode()  — autoregressive inference (argmax at each step)

Weight initialisation: Xavier uniform for all 2-D parameter matrices;
    bias vectors zero; embedding weights ~ N(0, 0.02) with PAD row zeroed.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Sequence-to-Sequence Transformer (§ 3).

    Args:
        src_vocab_size : Source vocabulary size.
        trg_vocab_size : Target vocabulary size.
        d_model        : Model dimensionality       (paper: 512).
        n_heads        : Number of attention heads  (paper: 8).
        n_layers       : Encoder / decoder depth    (paper: 6).
        d_ff           : Feed-forward inner dim     (paper: 2048).
        dropout        : Dropout probability        (paper: 0.1).
        max_seq_len    : Maximum sequence length.
        pad_token_id   : <PAD> token ID.
    """

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model:      int   = 512,
        n_heads:      int   = 8,
        n_layers:     int   = 6,
        d_ff:         int   = 2048,
        dropout:      float = 0.1,
        max_seq_len:  int   = 5000,
        pad_token_id: int   = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id: int = pad_token_id

        # ── Sub-networks ──────────────────────────────────────────────────────
        self.encoder: Encoder = Encoder(
            vocab_size   = src_vocab_size,
            d_model      = d_model,
            n_layers     = n_layers,
            n_heads      = n_heads,
            d_ff         = d_ff,
            dropout      = dropout,
            max_seq_len  = max_seq_len,
            pad_token_id = pad_token_id,
        )
        self.decoder: Decoder = Decoder(
            vocab_size   = trg_vocab_size,
            d_model      = d_model,
            n_layers     = n_layers,
            n_heads      = n_heads,
            d_ff         = d_ff,
            dropout      = dropout,
            max_seq_len  = max_seq_len,
            pad_token_id = pad_token_id,
        )

        # § 3: "a linear transformation and softmax function to produce [probabilities]"
        # bias=False keeps this a pure projection consistent with the paper
        self.output_proj: nn.Linear = nn.Linear(d_model, trg_vocab_size, bias=False)

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Xavier uniform for Linear weights (standard for Transformers);
        N(0, 0.02) for Embeddings with PAD row zeroed.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ── Mask helpers ──────────────────────────────────────────────────────────

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Build a padding mask for the encoder's self-attention.

        A position is masked (=1) wherever the source token is <PAD>.
        The extra two singleton dims let it broadcast over
        (batch, heads, T_q, T_src).

        Args:
            src  : (batch, T_src) integer token IDs

        Returns:
            mask : (batch, 1, 1, T_src)  — 1 at PAD, 0 elsewhere
        """
        return (src == self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Build a combined causal + padding mask for the decoder.

        § 3.2.3: "We … modify the self-attention sub-layer in the decoder
        stack to prevent positions from attending to subsequent positions."

        A position (q, k) is masked if:
            a) k > q  (causal: would peek at a future token), OR
            b) trg[k] == <PAD>  (structural padding)

        Args:
            trg  : (batch, T_trg) integer token IDs

        Returns:
            mask : (batch, 1, T_trg, T_trg)  — True at forbidden positions
        """
        T: int = trg.size(1)

        # Upper-triangular 1s above the diagonal = future positions
        # Shape: (1, 1, T, T) for broadcasting over (batch, heads, T_q, T_k)
        causal_mask: torch.Tensor = torch.triu(
            torch.ones(T, T, device=trg.device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)

        # Padding: 1 wherever the *key* position is <PAD>
        # Shape: (batch, 1, 1, T) broadcasts to (batch, 1, T_q, T)
        pad_mask: torch.Tensor = (trg == self.pad_token_id).unsqueeze(1).unsqueeze(2)

        # Union: mask if causal OR pad
        return causal_mask | pad_mask   # (batch, 1, T, T)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        src:      torch.Tensor,
        trg:      torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        trg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass (teacher-forced, for training).

        Args:
            src      : (batch, T_src) source token IDs
            trg      : (batch, T_trg) target token IDs, right-shifted (starts with <SOS>)
            src_mask : pre-built source mask (built automatically if None)
            trg_mask : pre-built target mask (built automatically if None)

        Returns:
            logits : (batch, T_trg, trg_vocab_size) un-normalised log-probabilities
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if trg_mask is None:
            trg_mask = self.make_trg_mask(trg)

        # Encode
        encoder_out, _enc_attn = self.encoder(src, src_mask)
        # Decode (cross-attends to encoder_out)
        decoder_out, _self_attn, _cross_attn = self.decoder(
            trg, encoder_out, trg_mask=trg_mask, src_mask=src_mask
        )
        # Project to vocabulary
        logits: torch.Tensor = self.output_proj(decoder_out)
        return logits   # (batch, T_trg, vocab_size)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        src:          torch.Tensor,
        sos_token_id: int,
        eos_token_id: int,
        max_len:      int = 100,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding (argmax at each step).

        Generates tokens one at a time until <EOS> is produced or
        max_len tokens have been generated.

        Args:
            src          : (1, T_src) single source sequence  [do not batch]
            sos_token_id : <SOS> token ID (decoder start token)
            eos_token_id : <EOS> token ID (generation stop condition)
            max_len      : guard against infinite loops

        Returns:
            generated : (1, T_out) generated token IDs (SOS excluded)
        """
        self.eval()
        device: torch.device = src.device

        src_mask: torch.Tensor = self.make_src_mask(src)
        encoder_out, _ = self.encoder(src, src_mask)

        # Initialise decoder input with <SOS>
        trg: torch.Tensor = torch.tensor(
            [[sos_token_id]], dtype=torch.long, device=device
        )

        for _ in range(max_len):
            trg_mask: torch.Tensor = self.make_trg_mask(trg)
            dec_out, _, _ = self.decoder(
                trg, encoder_out, trg_mask=trg_mask, src_mask=src_mask
            )
            # Logits for the last generated position
            next_logits: torch.Tensor = self.output_proj(dec_out[:, -1, :])
            next_token:  torch.Tensor = next_logits.argmax(dim=-1, keepdim=True)
            trg = torch.cat([trg, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return trg[:, 1:]   # strip leading <SOS>

