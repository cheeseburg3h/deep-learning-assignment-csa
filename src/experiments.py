"""Run experiments comparing activation functions and gradient descent variants."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .activations import ACTIVATIONS
from .data_utils import train_test_split_dataset
from .losses import binary_cross_entropy
from .network import FullyConnectedNetwork
from .trainer import Trainer


def run_experiments(
    dataset_path: str | Path,
    hidden_neurons: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test, _ = train_test_split_dataset(dataset_path)
    input_size = X_train.shape[1]
    results: List[Dict[str, float | str]] = []

    for activation_name, gd_variant in itertools.product(ACTIVATIONS.keys(), ["batch", "stochastic", "minibatch"]):
        network = FullyConnectedNetwork(
            layer_sizes=[input_size, hidden_neurons, 1],
            hidden_activation=activation_name,
            output_activation="sigmoid",
            weight_scale=0.1,
            seed=seed,
        )
        trainer = Trainer(
            network=network,
            loss=binary_cross_entropy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=epochs,
            gradient_descent=gd_variant,
        )
        history = trainer.fit(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        results.append(
            {
                "activation": activation_name,
                "gd_variant": gd_variant,
                "final_train_loss": history.losses[-1],
                "final_train_accuracy": history.accuracies[-1],
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stroke prediction neural network experiments")
    parser.add_argument(
        "--data",
        type=str,
        default="data/healthcare-dataset-stroke-data.csv",
        help="Path to the stroke dataset CSV file.",
    )
    parser.add_argument("--hidden", type=int, default=32, help="Number of neurons in the hidden layer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for gradient descent.")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size.")
    args = parser.parse_args()

    df_results = run_experiments(
        dataset_path=args.data,
        hidden_neurons=args.hidden,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    print(df_results.to_string(index=False))
