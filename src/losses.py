"""Loss functions for supervised learning."""

from __future__ import annotations

import numpy as np


class LossFunction:
    """Container for a loss function and its derivative."""

    def __init__(self, func, derivative, name: str):
        self.func = func
        self.derivative = derivative
        self.name = name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self.func(y_true, y_pred))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.derivative(y_true, y_pred)


_DEF_EPS = 1e-12


def _clip(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, _DEF_EPS, 1 - _DEF_EPS)


binary_cross_entropy = LossFunction(
    func=lambda y_true, y_pred: -np.mean(
        y_true * np.log(_clip(y_pred)) + (1 - y_true) * np.log(_clip(1 - y_pred))
    ),
    derivative=lambda y_true, y_pred: (-(y_true / _clip(y_pred)) + (1 - y_true) / _clip(1 - y_pred)) / y_true.shape[0],
    name="binary_cross_entropy",
)

LOSSES = {
    "binary_cross_entropy": binary_cross_entropy,
}


def get_loss(name: str) -> LossFunction:
    lowered = name.lower()
    if lowered not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[lowered]
