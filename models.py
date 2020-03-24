from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import List


def train_logreg(X: np.ndarray, y: List[int]) -> LogisticRegression:
    """Train a logistic regression model on the given data."""
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf


