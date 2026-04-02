"""
config.py
==========
All hyperparameters for the Transformer model, using a single typed
dataclass.  Every value is annotated with its paper source and the
original "base model" default from Table 3 of:

    Vaswani et al. (2017) — "Attention Is All You Need"

Usage:
    from config import TransformerConfig
    cfg = TransformerConfig()          # all defaults
    cfg = TransformerConfig(d_model=512, n_layers=6)   # override
"""

import torch
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    """
    Single flat config for the full Tiny-Shakespeare Transformer.

    Paper defaults (base model) are noted in comments.
    We use smaller defaults so the model trains on CPU in reasonable time.
    """

    # ── Vocabulary & sequence ─────────────────────────────────────────────────
    # Set automatically by the CharTokenizer; overridden in main.py
    vocab_size:   int = 100        # updated from tokenizer at runtime
    max_seq_len:  int = 256        # maximum sequence length for PE table

    # ── Model dimensions (§ 3.1) ─────────────────────────────────────────────
    d_model:   int = 256           # model dimensionality          (paper: 512)
    n_heads:   int = 8             # number of attention heads     (paper: 8)
    n_layers:  int = 4             # encoder/decoder stack depth   (paper: 6)
    d_ff:      int = 1024          # feed-forward inner dim        (paper: 2048 = 4×d_model)

    # d_k = d_v = d_model // n_heads (§ 3.2.2); derived in __post_init__
    d_k: int = field(init=False)
    d_v: int = field(init=False)

    # ── Regularisation (§ 5.4) ───────────────────────────────────────────────
    dropout:          float = 0.1  # dropout probability           (paper: 0.1)
    label_smoothing:  float = 0.1  # label-smoothing ε             (paper: 0.1)

    # ── Training (§ 5.3) ─────────────────────────────────────────────────────
    batch_size:    int   = 64      # mini-batch size
    num_epochs:    int   = 20      # training epochs
    warmup_steps:  int   = 4000   # LR warmup steps               (paper: 4000)
    clip_norm:     float = 1.0    # gradient-clipping max norm

    # Adam hyper-parameters (§ 5.3)
    adam_beta1:  float = 0.9
    adam_beta2:  float = 0.98
    adam_eps:    float = 1e-9

    # ── Special tokens ────────────────────────────────────────────────────────
    pad_token_id: int = 0          # <PAD>
    sos_token_id: int = 1          # <SOS>  (decoder primer)
    eos_token_id: int = 2          # <EOS>

    # ── Data ─────────────────────────────────────────────────────────────────
    chunk_size:   int   = 128      # character chunk length L
    train_split:  float = 0.9      # fraction of corpus used for training

    # ── Logging & checkpointing ───────────────────────────────────────────────
    log_interval: int = 50         # print loss every N batches
    save_dir:     str = "checkpoints"

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self) -> None:
        """Validate constraints and derive dependent fields."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        # § 3.2.2: d_k = d_v = d_model / h  (derived, not user-defined)
        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_heads

