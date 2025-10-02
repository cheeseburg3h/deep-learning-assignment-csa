# Deep Learning Implementation Assignment

This project contains a implementation of a small feedforward neural network for predicting strokes using the Kaggle healthcare dataset.

## Project structure

```
src/
├── activations.py     # Activation functions (sigmoid, ReLU, tanh) and derivatives
├── data_utils.py      # Dataset loading, preprocessing, and 80/20 train-test split
├── experiments.py     # Script to run experiments across activations & GD variants
├── losses.py          # Binary cross-entropy loss and derivative
├── metrics.py         # Accuracy metric for evaluation
├── network.py         # Fully-connected network with forward & backward passes
└── trainer.py         # Training loop supporting batch, stochastic & minibatch GD
```

## Running experiments

Execute the experiment script to compare activation functions and gradient descent variants:

```bash
python -m src.experiments --data data/healthcare-dataset-stroke-data.csv --epochs 100 --hidden 64 --lr 0.01 --batch-size 32
```
