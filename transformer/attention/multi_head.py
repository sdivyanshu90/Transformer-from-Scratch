"""
transformer/attention/multi_head.py
=====================================
§ 3.2.2 — Multi-Head Attention

"Instead of performing a single attention function with d_model-dimensional
keys, values and queries, we found it beneficial to linearly project the
queries, keys and values h times with different, learned linear projections
to d_k, d_k and d_v dimensions respectively.  … the h heads are concatenated
and once again projected, resulting in the final values."
                                                    — Vaswani et al. (2017)

    MultiHead(Q,K,V) = Concat(head₁, …, headₕ) Wᴼ
    where  headᵢ = Attention(Q Wᵢᴼ, K Wᵢᴷ, V Wᵢᵛ)

All h heads are computed in parallel via a batched matrix multiply: the
single W_Q / W_K / W_V projections (size d_model × d_model) simultaneously
encode all head projections packed together.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .scaled_dot_product import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (§ 3.2.2).

    Args:
        d_model  : Total model dimensionality (e.g. 512).
        n_heads  : Number of parallel attention heads h (e.g. 8).
        dropout  : Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model:  int = d_model
        self.n_heads:  int = n_heads
        # § 3.2.2: d_k = d_v = d_model / h
        self.d_k: int = d_model // n_heads
        self.d_v: int = d_model // n_heads

        # Learned projection matrices — no bias (as in the paper)
        # Each matrix is d_model × d_model; internally packs all h heads
        self.W_Q: nn.Linear = nn.Linear(d_model, d_model, bias=False)  # Wᴼ stacked
        self.W_K: nn.Linear = nn.Linear(d_model, d_model, bias=False)  # Wᴷ stacked
        self.W_V: nn.Linear = nn.Linear(d_model, d_model, bias=False)  # Wᵛ stacked
        # Output projection W_O: h·d_v → d_model
        self.W_O: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Store last attention weights for visualisation / debugging
        self.last_attn_weights: Optional[torch.Tensor] = None

    # ── Shape helpers ─────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, T, d_model) → (batch, n_heads, T, d_k).

        Splits the last dimension into (n_heads, d_k) and transposes to put
        the head dimension second so all heads process independently.
        """
        B, T, _ = x.size()
        # (B, T, n_heads, d_k)
        x = x.view(B, T, self.n_heads, self.d_k)
        # (B, n_heads, T, d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse _split_heads: (batch, n_heads, T, d_k) → (batch, T, d_model).

        Transposes back and reshapes to concatenate all head outputs.
        """
        B, _, T, _ = x.size()
        # (B, T, n_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # (B, T, d_model)
        return x.view(B, T, self.d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        query:  torch.Tensor,
        key:    torch.Tensor,
        value:  torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.

        Args:
            query : (batch, T_q, d_model)
            key   : (batch, T_k, d_model)
            value : (batch, T_k, d_model)
            mask  : optional mask broadcastable to
                    (batch, n_heads, T_q, T_k)
                    — True at positions to ignore.

        Returns:
            output  : (batch, T_q, d_model)
            weights : (batch, n_heads, T_q, T_k)  attention weights
        """
        # 1. Linear projections W_Q, W_K, W_V
        Q: torch.Tensor = self.W_Q(query)   # (B, T_q, d_model)
        K: torch.Tensor = self.W_K(key)     # (B, T_k, d_model)
        V: torch.Tensor = self.W_V(value)   # (B, T_k, d_model)

        # 2. Split into h heads
        Q = self._split_heads(Q)  # (B, h, T_q, d_k)
        K = self._split_heads(K)  # (B, h, T_k, d_k)
        V = self._split_heads(V)  # (B, h, T_k, d_v)

        # 3. Expand 3-D mask to 4-D if needed (add head dim)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)          # (B, 1, T_q, T_k)

        # 4. Scaled dot-product attention — all heads in parallel
        attn_out, weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        self.last_attn_weights = weights.detach()

        # 5. Concatenate + final projection W_O
        attn_out = self._merge_heads(attn_out)          # (B, T_q, d_model)
        output: torch.Tensor = self.W_O(attn_out)       # (B, T_q, d_model)

        return output, weights

