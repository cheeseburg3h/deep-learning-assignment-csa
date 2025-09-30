# Stroke Prediction Neural Network Assignment

This project contains a from-scratch implementation of a small feedforward neural network for
predicting strokes using the Kaggle healthcare dataset. The code is organised into reusable
modules that implement activation functions, loss, forward and backward passes, gradient descent
variants, and experiment tooling required by the assignment brief.

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

## Prerequisites

Install the required Python packages (NumPy, pandas, and scikit-learn):

```bash
pip install numpy pandas scikit-learn
```

Download the dataset from Kaggle and place it at `data/healthcare-dataset-stroke-data.csv`:

```bash
mkdir -p data
# Copy the downloaded CSV into the data/ directory
```

## Running experiments

Execute the experiment script to compare activation functions and gradient descent variants:

```bash
python -m src.experiments --data data/healthcare-dataset-stroke-data.csv --epochs 100 --hidden 64 --lr 0.01 --batch-size 32
```

The script prints a table summarising the training and test metrics for each combination of
activation function (`sigmoid`, `relu`, `tanh`) and gradient descent variant (`batch`, `stochastic`,
`minibatch`). Adjust the hyperparameters via the command-line flags as needed.

## Custom usage

The building blocks can be combined for other experiments. For instance, you can instantiate the
`FullyConnectedNetwork` and `Trainer` classes directly to explore custom network depths, learning
rates, or evaluation strategies.
