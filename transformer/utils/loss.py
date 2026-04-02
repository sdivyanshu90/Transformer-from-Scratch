"""
transformer/utils/loss.py
==========================
§ 5.4 — Label Smoothing

"We employed label smoothing of value ε_ls = 0.1.  This hurts perplexity,
as the model learns to be more unsure, but improves accuracy and BLEU score."
                                                    — Vaswani et al. (2017)

Standard one-hot cross-entropy trains the model to assign all probability to
the correct token, which leads to overconfidence.  Label smoothing replaces
the hard target distribution with a soft mixture:

    q(k|x) = (1 − ε)   if k == target
    q(k|x) = ε / (V−1)  otherwise    [V = vocab size]

PAD positions are excluded from both the distribution and the loss average.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing (§ 5.4).

    Args:
        vocab_size   : Output vocabulary size V.
        pad_token_id : <PAD> token ID — excluded from loss and distribution.
        smoothing    : Label-smoothing factor ε (paper default: 0.1).
    """

    def __init__(
        self,
        vocab_size:   int,
        pad_token_id: int   = 0,
        smoothing:    float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size:  int   = vocab_size
        self.pad_token_id: int  = pad_token_id
        self.smoothing:   float = smoothing
        self.confidence:  float = 1.0 - smoothing   # mass on the true class

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy.

        Accepts either:
            logits  : (batch, T, V)  +  targets : (batch, T)   (3-D input)
          or
            logits  : (N, V)         +  targets : (N,)          (2-D input)

        Returns:
            loss : scalar — mean over non-PAD tokens
        """
        # Flatten to (N, V) and (N,) if needed
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits  = logits.reshape(B * T, V)
            targets = targets.reshape(B * T)

        # Log-probabilities for KL computation
        log_probs: torch.Tensor = F.log_softmax(logits, dim=-1)

        # ── Construct soft target distribution ────────────────────────────────
        # Start with uniform smoothing mass: ε / (V − 1) everywhere
        smooth_dist: torch.Tensor = torch.full_like(
            log_probs, self.smoothing / (self.vocab_size - 1)
        )
        # Place confidence (1 − ε) on the correct token
        smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # ── Zero out PAD rows entirely ─────────────────────────────────────────
        pad_mask: torch.Tensor = (targets == self.pad_token_id)
        smooth_dist[pad_mask] = 0.0

        # ── KL-divergence loss = − Σ q(k) · log p(k) ─────────────────────────
        loss: torch.Tensor = -(smooth_dist * log_probs).sum(dim=-1)

        # Mask PAD positions and average over non-PAD tokens only
        loss = loss.masked_fill(pad_mask, 0.0)
        n_tokens: torch.Tensor = (~pad_mask).sum().float().clamp(min=1.0)
        return loss.sum() / n_tokens

