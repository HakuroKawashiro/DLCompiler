"""Learning rate schedulers."""

import math
from typing import Any, Dict, List

from .optim import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers.

    Attributes:
        optimizer: The optimizer whose learning rate to adjust
        base_lrs: Initial learning rates copied from optimizer
        last_epoch: The index of the last epoch
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lrs = [optimizer.lr] if hasattr(optimizer, 'lr') else []
        self.last_epoch = -1

    def step(self) -> None:
        """Advance to the next epoch and update learning rate."""
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch. Override in subclasses."""
        raise NotImplementedError()

    def get_last_lr(self) -> List[float]:
        """Return the last computed learning rate(s)."""
        return [self.optimizer.lr] if hasattr(self.optimizer, 'lr') else []

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for serialization."""
        return {
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Restore scheduler state from serialization."""
        self.base_lrs = d.get('base_lrs', self.base_lrs)
        self.last_epoch = d.get('last_epoch', self.last_epoch)


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs.

    Args:
        optimizer: Wrapped optimizer
        step_size: Number of epochs between lr decays
        gamma: Multiplicative decay factor (default 0.1)
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        if (self.last_epoch + 1) % self.step_size == 0:
            decay_factor = self.gamma ** ((self.last_epoch + 1) // self.step_size)
            return self.base_lrs[0] * decay_factor
        return self.optimizer.lr


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler.

    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))

    Args:
        optimizer: Wrapped optimizer
        T_max: Number of epochs for one full cosine cycle
        eta_min: Minimum learning rate (default 0.0)
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        epoch = self.last_epoch + 1
        lr = self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * (
            1.0 + math.cos(math.pi * epoch / self.T_max)
        )
        return max(lr, self.eta_min)
