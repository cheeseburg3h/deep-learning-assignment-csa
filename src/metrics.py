"""Evaluation metrics for model assessment."""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predictions = (y_pred >= 0.5).astype(int)
    return float(np.mean(predictions == y_true))
