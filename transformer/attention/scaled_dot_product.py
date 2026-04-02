"""
transformer/attention/scaled_dot_product.py
============================================
§ 3.2.1 — Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V

"We call our particular attention 'Scaled Dot-Product Attention'.
The input consists of queries and keys of dimension dₖ, and values
of dimension dᵥ.  We compute the dot products of the query with all
keys, divide each by √dₖ, and apply a softmax function to obtain
the weights on the values."                       — Vaswani et al. (2017)

The 1/√dₖ scaling prevents the dot products from growing large in
magnitude and pushing the softmax into regions of very small gradients.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query:   torch.Tensor,
    key:     torch.Tensor,
    value:   torch.Tensor,
    mask:    Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout]  = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention (§ 3.2.1).

    Args:
        query   : (batch, heads, T_q, d_k)
        key     : (batch, heads, T_k, d_k)
        value   : (batch, heads, T_k, d_v)   [T_k == T_v always]
        mask    : Boolean tensor broadcastable to
                  (batch, heads, T_q, T_k).
                  True (1) at a position means that position is *masked out*
                  (set to −∞ before softmax), i.e. the model cannot attend there.
        dropout : Optional nn.Dropout applied to attention weights after softmax.

    Returns:
        output  : (batch, heads, T_q, d_v) — weighted sum of values
        weights : (batch, heads, T_q, T_k) — post-softmax attention weights
    """
    d_k: int = query.size(-1)

    # ── Step 1: raw attention scores  QKᵀ / √dₖ ─────────────────────────────
    # Shape: (batch, heads, T_q, T_k)
    scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # ── Step 2: apply mask (padding / causal) ────────────────────────────────
    # § 3.2.3: "We … set to −∞ all values … which correspond to illegal connections."
    if mask is not None:
        scores = scores.masked_fill(mask.bool(), float("-inf"))

    # ── Step 3: softmax over the key dimension ────────────────────────────────
    # Shape: (batch, heads, T_q, T_k)
    weights: torch.Tensor = F.softmax(scores, dim=-1)

    # Guard against NaN that arises when an entire row is masked (all −∞ → NaN)
    weights = torch.nan_to_num(weights, nan=0.0)

    # ── Step 4: optional dropout on attention weights ─────────────────────────
    if dropout is not None:
        weights = dropout(weights)

    # ── Step 5: weighted sum over values ──────────────────────────────────────
    # Shape: (batch, heads, T_q, d_v)
    output: torch.Tensor = torch.matmul(weights, value)

    return output, weights

