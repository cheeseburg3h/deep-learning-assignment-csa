"""Implementation of a simple fully-connected neural network from scratch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .activations import ActivationFunction, get_activation


@dataclass
class Layer:
    weights: np.ndarray
    biases: np.ndarray
    activation: ActivationFunction

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = x @ self.weights + self.biases
        a = self.activation(z)
        return z, a


class FullyConnectedNetwork:
    """A minimal dense neural network with one or more hidden layers."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        weight_scale: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must specify at least input and output size")
        self.rng = np.random.default_rng(seed)
        self.layers: List[Layer] = []
        for idx in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[idx], layer_sizes[idx + 1]
            activation_name = output_activation if idx == len(layer_sizes) - 2 else hidden_activation
            activation = get_activation(activation_name)
            limit = weight_scale
            weights = self.rng.normal(0.0, limit, size=(fan_in, fan_out))
            biases = np.zeros(fan_out)
            self.layers.append(Layer(weights=weights, biases=biases, activation=activation))

    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        zs = []
        for layer in self.layers:
            z, a = layer.forward(activations[-1])
            zs.append(z)
            activations.append(a)
        return zs, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, activations = self.forward(x)
        return activations[-1]

    def backward(
        self,
        loss_grad: np.ndarray,
        zs: Sequence[np.ndarray],
        activations: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        grad_weights: List[np.ndarray] = [np.zeros_like(layer.weights) for layer in self.layers]
        grad_biases: List[np.ndarray] = [np.zeros_like(layer.biases) for layer in self.layers]

        delta = loss_grad
        for idx in reversed(range(len(self.layers))):
            z = zs[idx]
            activation_grad = self.layers[idx].activation.grad(z)
            delta = delta * activation_grad
            grad_weights[idx] = activations[idx].T @ delta
            grad_biases[idx] = np.sum(delta, axis=0)
            if idx > 0:
                delta = delta @ self.layers[idx].weights.T
        return grad_weights, grad_biases

    def update_parameters(self, grad_weights: Sequence[np.ndarray], grad_biases: Sequence[np.ndarray], lr: float) -> None:
        for layer, dW, dB in zip(self.layers, grad_weights, grad_biases):
            layer.weights -= lr * dW
            layer.biases -= lr * dB
