"""
training/trainer.py
====================
Training loop for the Tiny-Shakespeare Transformer.

Provides three composable functions:
    train_epoch  — one full pass over the training DataLoader
    evaluate     — validation pass (no grad, returns avg loss)
    fit          — outer loop over epochs; checkpoints best model

Design notes:
  • Gradients are zeroed *before* each forward pass (standard practice).
  • Gradient clipping (max-norm = 1.0) stabilises early training.
  • The WarmupScheduler calls optimizer.step() internally, so we must NOT
    call optimizer.step() separately.
  • A checkpoint is saved whenever validation loss reaches a new minimum.

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""

import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer.model.transformer import Transformer
from transformer.utils.loss import LabelSmoothingLoss
from transformer.utils.scheduler import WarmupScheduler


# ── Single training epoch ─────────────────────────────────────────────────────

def train_epoch(
    model:        Transformer,
    loader:       DataLoader,
    criterion:    LabelSmoothingLoss,
    scheduler:    WarmupScheduler,
    device:       torch.device,
    log_interval: int = 50,
) -> float:
    """
    Run one full training epoch.

    Args:
        model        : Transformer model in training mode.
        loader       : Training DataLoader yielding (src, trg, labels).
        criterion    : Label-smoothed cross-entropy loss.
        scheduler    : WarmupScheduler; its step() updates LR and optimizer.
        device       : Compute device.
        log_interval : Print running loss every N batches.

    Returns:
        avg_loss : Mean label-smoothed loss over all batches.
    """
    model.train()
    total_loss: float = 0.0
    n_batches:  int   = 0
    t0: float = time.time()

    for batch_idx, (src, trg, labels) in enumerate(loader):
        src    = src.to(device,    non_blocking=True)   # (B, T)
        trg    = trg.to(device,    non_blocking=True)   # (B, T)
        labels = labels.to(device, non_blocking=True)   # (B, T)

        # ── Forward ──────────────────────────────────────────────────────────
        logits: torch.Tensor = model(src, trg)          # (B, T, vocab)

        # ── Loss (handles flattening internally) ─────────────────────────────
        loss: torch.Tensor = criterion(logits, labels)

        # ── Backward ─────────────────────────────────────────────────────────
        scheduler.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stabilises training; common in Transformer work)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # WarmupScheduler.step() updates LR and calls optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed: float = time.time() - t0
            avg: float = total_loss / n_batches
            print(
                f"  [{batch_idx + 1:5d}/{len(loader):5d}]  "
                f"loss={avg:.4f}  "
                f"lr={scheduler.get_last_lr():.2e}  "
                f"step={scheduler.current_step:6d}  "
                f"({elapsed:.1f}s)"
            )

    return total_loss / max(n_batches, 1)


# ── Validation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     Transformer,
    loader:    DataLoader,
    criterion: LabelSmoothingLoss,
    device:    torch.device,
) -> float:
    """
    Compute average validation loss (no gradient updates).

    Args:
        model     : Transformer model.
        loader    : Validation DataLoader.
        criterion : Label-smoothed cross-entropy loss.
        device    : Compute device.

    Returns:
        avg_loss : Mean loss over all validation batches.
    """
    model.eval()
    total_loss: float = 0.0
    n_batches:  int   = 0

    for src, trg, labels in loader:
        src    = src.to(device,    non_blocking=True)
        trg    = trg.to(device,    non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits: torch.Tensor = model(src, trg)
        loss:   torch.Tensor = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Outer fit loop ────────────────────────────────────────────────────────────

def fit(
    model:        Transformer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    criterion:    LabelSmoothingLoss,
    scheduler:    WarmupScheduler,
    num_epochs:   int,
    device:       torch.device,
    save_dir:     str = "checkpoints",
    log_interval: int = 50,
) -> Dict[str, List[float]]:
    """
    Full training run: iterates over epochs, evaluates, and checkpoints.

    Saves 'best_model.pt' whenever the validation loss reaches a new minimum.

    Args:
        model        : Transformer model.
        train_loader : Training DataLoader.
        val_loader   : Validation DataLoader.
        criterion    : Label-smoothed cross-entropy loss.
        scheduler    : WarmupScheduler.
        num_epochs   : Total epochs to train.
        device       : Compute device.
        save_dir     : Directory for checkpoints.
        log_interval : Batch-level print frequency.

    Returns:
        history : {"train_loss": [...], "val_loss": [...]}
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val: float = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Training on {device}  |  epochs={num_epochs}  |  "
          f"batches/epoch={len(train_loader)}")
    print(sep)

    for epoch in range(1, num_epochs + 1):
        t_epoch: float = time.time()
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        # ── Train ──────────────────────────────────────────────────────────
        train_loss: float = train_epoch(
            model, train_loader, criterion, scheduler, device, log_interval
        )
        history["train_loss"].append(train_loss)

        # ── Validate ───────────────────────────────────────────────────────
        val_loss: float = evaluate(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)

        elapsed_epoch: float = time.time() - t_epoch
        print(
            f"\n  Epoch {epoch:3d} summary  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"({elapsed_epoch:.1f}s)"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path: str = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": scheduler.optimizer.state_dict(),
                    "val_loss":            val_loss,
                    "scheduler_step":      scheduler.current_step,
                },
                ckpt_path,
            )
            print(f"  ✓ New best saved → {ckpt_path}  (val_loss={val_loss:.4f})")

    print(f"\n{sep}")
    print(f"  Training complete.  Best val_loss = {best_val:.4f}")
    print(sep)
    return history

