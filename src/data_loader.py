"""Data loading helpers for the project."""

from typing import Tuple
import pandas as pd


def load_creditcard_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the credit card dataset and return features and target.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the dataset. The CSV must contain a
        column named ``Class`` which will be used as the target variable.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix (all columns except ``Class``).
    y : pandas.Series
        Target vector (the ``Class`` column).
    """
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y