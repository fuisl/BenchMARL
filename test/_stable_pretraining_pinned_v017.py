"""Frozen scheduler fixture from stable-pretraining v0.1.7.

Source commit ``bce7c8b35a62399d0529068ed7fea5dd2ce9021e``,
``stable_pretraining/optim/lr_scheduler.py``.  This was the newest published
release when LeWM commit 8edfeb33 was created.  LeWM did not ship a lock file,
so the date-based resolution is recorded rather than presented as an
environment lock. MIT license: ``test/STABLE_PRETRAINING_LICENSE``.
"""

import math

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                (
                    self.warmup_start_lr
                    + (base_lr - self.warmup_start_lr)
                    * self.last_epoch
                    / self.warmup_steps
                )
                for base_lr in self.base_lrs
            ]
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps)
                    / (self.max_steps - self.warmup_steps)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]
