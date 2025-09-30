"""Training utilities implementing various gradient descent variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .losses import LossFunction
from .metrics import accuracy_score
from .network import FullyConnectedNetwork


@dataclass
class TrainingHistory:
    losses: List[float]
    accuracies: List[float]


class Trainer:
    def __init__(
        self,
        network: FullyConnectedNetwork,
        loss: LossFunction,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 100,
        gradient_descent: str = "batch",
        shuffle: bool = True,
    ) -> None:
        self.network = network
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.gradient_descent = gradient_descent
        self.shuffle = shuffle

    def _iter_batches(self, X: np.ndarray, y: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.gradient_descent == "batch":
            yield X[indices], y[indices]
        elif self.gradient_descent == "stochastic":
            for idx in indices:
                yield X[idx : idx + 1], y[idx : idx + 1]
        elif self.gradient_descent == "minibatch":
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                yield X[batch_idx], y[batch_idx]
        else:
            raise ValueError(
                "gradient_descent must be one of {'batch', 'stochastic', 'minibatch'}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> TrainingHistory:
        losses: List[float] = []
        accuracies: List[float] = []
        for epoch in range(self.max_epochs):
            for X_batch, y_batch in self._iter_batches(X, y):
                zs, activations = self.network.forward(X_batch)
                y_pred = activations[-1]
                loss_grad = self.loss.grad(y_batch, y_pred)
                grad_weights, grad_biases = self.network.backward(loss_grad, zs, activations)
                self.network.update_parameters(grad_weights, grad_biases, self.learning_rate)
            eval_X = X if X_val is None else X_val
            eval_y = y if y_val is None else y_val
            _, activations_eval = self.network.forward(eval_X)
            predictions = activations_eval[-1]
            loss_value = self.loss(eval_y, predictions)
            acc_value = accuracy_score(eval_y, predictions)
            losses.append(loss_value)
            accuracies.append(acc_value)
        return TrainingHistory(losses=losses, accuracies=accuracies)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        _, activations = self.network.forward(X)
        predictions = activations[-1]
        return {
            "loss": self.loss(y, predictions),
            "accuracy": accuracy_score(y, predictions),
        }
