"""
main.py
========
Entry point — wires the full Tiny-Shakespeare Transformer pipeline together.

Pipeline:
    1.  Parse / build  TransformerConfig
    2.  Download data  (auto-cached) + build CharTokenizer
    3.  Instantiate    Transformer model
    4.  Create         LabelSmoothingLoss   (§ 5.4)
    5.  Create         Adam optimizer       (§ 5.3)
    6.  Create         WarmupScheduler      (§ 5.3)
    7.  fit()          training loop with checkpointing
    8.  Evaluate       token-level accuracy on the validation set
    9.  Generate       Shakespeare-like text from example prompts
   10.  Reload best checkpoint and re-generate

Run:
    python main.py

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import os
import torch

from config import TransformerConfig
from data.dataset import get_dataloaders
from transformer.model.transformer import Transformer
from transformer.utils.loss import LabelSmoothingLoss
from transformer.utils.scheduler import WarmupScheduler
from training.trainer import fit
from training.evaluator import compute_accuracy, run_demo_generation, model_summary


def main() -> None:
    """Full training and evaluation pipeline."""

    # ── 1. Configuration ──────────────────────────────────────────────────────
    cfg: TransformerConfig = TransformerConfig()
    device: torch.device   = torch.device(cfg.device)
    print(f"\nDevice  : {device}")
    print(f"d_model : {cfg.d_model}   n_heads : {cfg.n_heads}   "
          f"n_layers : {cfg.n_layers}   d_ff : {cfg.d_ff}")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    print("\n[data] Loading Tiny Shakespeare …")
    train_loader, val_loader, tokenizer = get_dataloaders(
        chunk_size  = cfg.chunk_size,
        batch_size  = cfg.batch_size,
        train_frac  = cfg.train_split,
    )

    # Patch vocab_size into config now that the tokenizer is ready
    cfg.vocab_size = tokenizer.vocab_size
    print(f"[data] vocab_size={cfg.vocab_size}  "
          f"train_batches={len(train_loader)}  "
          f"val_batches={len(val_loader)}")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model: Transformer = Transformer(
        src_vocab_size = cfg.vocab_size,
        trg_vocab_size = cfg.vocab_size,
        d_model        = cfg.d_model,
        n_heads        = cfg.n_heads,
        n_layers       = cfg.n_layers,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        max_seq_len    = cfg.max_seq_len,
        pad_token_id   = cfg.pad_token_id,
    ).to(device)

    model_summary(model)

    # ── 4. Loss (§ 5.4) ───────────────────────────────────────────────────────
    criterion: LabelSmoothingLoss = LabelSmoothingLoss(
        vocab_size   = cfg.vocab_size,
        pad_token_id = cfg.pad_token_id,
        smoothing    = cfg.label_smoothing,
    )

    # ── 5 & 6. Adam + WarmupScheduler (§ 5.3) ────────────────────────────────
    # lr=1.0 so WarmupScheduler has full multiplicative control
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = 1.0,
        betas = (cfg.adam_beta1, cfg.adam_beta2),
        eps   = cfg.adam_eps,
    )
    scheduler: WarmupScheduler = WarmupScheduler(
        optimizer    = optimizer,
        d_model      = cfg.d_model,
        warmup_steps = cfg.warmup_steps,
    )

    # ── 7. Fit ────────────────────────────────────────────────────────────────
    history = fit(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = criterion,
        scheduler    = scheduler,
        num_epochs   = cfg.num_epochs,
        device       = device,
        save_dir     = cfg.save_dir,
        log_interval = cfg.log_interval,
    )

    # ── 8. Accuracy ───────────────────────────────────────────────────────────
    print("\n[eval] Computing validation token accuracy …")
    val_acc: float = compute_accuracy(
        model, val_loader, device, cfg.pad_token_id
    )
    print(f"[eval] Validation token accuracy : {val_acc * 100:.2f}%")

    # ── 9. Text generation with last checkpoint ───────────────────────────────
    run_demo_generation(
        model       = model,
        tokenizer   = tokenizer,
        device      = device,
        temperature = 0.8,
        top_k       = 10,
    )

    # ── 10. Reload best checkpoint and re-generate ────────────────────────────
    best_ckpt: str = os.path.join(cfg.save_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        print(f"\n[main] Loading best checkpoint: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"[main] Best epoch={ckpt['epoch']}  "
            f"val_loss={ckpt['val_loss']:.4f}"
        )
        print("\n[main] Generating with best model:")
        run_demo_generation(
            model       = model,
            tokenizer   = tokenizer,
            device      = device,
            temperature = 0.8,
            top_k       = 10,
        )


if __name__ == "__main__":
    main()

