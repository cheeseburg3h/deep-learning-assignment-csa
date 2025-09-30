"""Utility helpers for loading and preprocessing the stroke dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


CATEGORICAL_COLUMNS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
TARGET_COLUMN = "stroke"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'healthcare-dataset-stroke-data.csv' "
            "from Kaggle and place it at this location."
        )
    return pd.read_csv(path)


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()
    df = df.dropna()
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN].astype(int).to_numpy().reshape(-1, 1)
    features = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, drop_first=True)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features.astype(float), target.astype(float), list(features.columns)


def train_test_split_dataset(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = load_dataset(csv_path)
    X, y, feature_names = preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_names
