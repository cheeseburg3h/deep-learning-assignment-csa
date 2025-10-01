"""Activation functions and their derivatives for neural network training."""

from __future__ import annotations

import numpy as np


class ActivationFunction:
    """Container for an activation function and its derivative."""

    def __init__(self, func, derivative, name: str):
        self.func = func
        self.derivative = derivative
        self.name = name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.func(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.derivative(x)


sigmoid = ActivationFunction(
    func=lambda x: 1.0 / (1.0 + np.exp(-x)),
    derivative=lambda x: sigmoid.func(x) * (1 - sigmoid.func(x)),
    name="sigmoid",
)

relu = ActivationFunction(
    func=lambda x: np.maximum(0.0, x),
    derivative=lambda x: (x > 0).astype(float),
    name="relu",
)

tanh = ActivationFunction(
    func=np.tanh,
    derivative=lambda x: 1.0 - np.tanh(x) ** 2,
    name="tanh",
)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
}


def get_activation(name: str) -> ActivationFunction:
    """Return an :class:`ActivationFunction` by name."""

    lowered = name.lower()
    if lowered not in ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}'. Available: {list(ACTIVATIONS)}")
    return ACTIVATIONS[lowered]
