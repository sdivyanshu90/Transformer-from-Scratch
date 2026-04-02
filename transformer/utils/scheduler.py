"""
transformer/utils/scheduler.py
================================
§ 5.3 — Optimizer and Learning-Rate Schedule

Optimizer (§ 5.3):
    Adam with  β₁ = 0.9,  β₂ = 0.98,  ε = 10⁻⁹

Learning-rate formula (§ 5.3):
    lrate = d_model^{−0.5} · min(step^{−0.5},  step · warmup_steps^{−1.5})

This linearly increases for the first `warmup_steps` training steps,
then decreases proportionally to the inverse square root of the step:

    • For step ≤ warmup_steps:  lrate ∝ step        (linear warm-up)
    • For step > warmup_steps:  lrate ∝ step^{−0.5} (inverse-sqrt decay)

The paper uses warmup_steps = 4000.

Usage:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)

    # Inside the training loop:
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scheduler.step()   # updates LR and calls optimizer.step()

Reference: Vaswani et al. (2017), "Attention Is All You Need"
"""


class WarmupScheduler:
    """
    Warmup Learning-Rate Scheduler (§ 5.3).

    Wraps a PyTorch optimizer and updates its learning rate at every call
    to step() according to the paper's formula.  The optimizer's initial lr
    should be set to 1.0 so the scheduler has full multiplicative control.

    Args:
        optimizer     : PyTorch optimizer (Adam recommended).
        d_model       : Model dimensionality (controls LR magnitude).
        warmup_steps  : Number of warm-up steps (paper default: 4000).
    """

    def __init__(
        self,
        optimizer,
        d_model:      int = 512,
        warmup_steps: int = 4000,
    ) -> None:
        self.optimizer    = optimizer
        self.d_model:     int   = d_model
        self.warmup_steps: int  = warmup_steps
        self._step:        int  = 0
        self._rate:        float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self) -> None:
        """
        Advance one training step:
            1. Increment step counter.
            2. Recompute learning rate using paper formula.
            3. Update all optimizer param-group LRs.
            4. Call optimizer.step().
        """
        self._step += 1
        self._rate = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._rate
        self.optimizer.step()

    def get_last_lr(self) -> float:
        """Return the learning rate used in the most recent step."""
        return self._rate

    @property
    def current_step(self) -> int:
        """Current training step (1-indexed)."""
        return self._step

    # ── Formula implementation ────────────────────────────────────────────────

    def _compute_lr(self, step: int) -> float:
        """
        § 5.3 formula:
            lrate = d_model^{−0.5} · min(step^{−0.5}, step · warmup_steps^{−1.5})

        Args:
            step : Training step number (must be ≥ 1).

        Returns:
            Learning rate as a float.
        """
        step = max(step, 1)   # guard against step=0
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

