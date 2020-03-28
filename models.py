from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import List
import pandas as pd
import os
from constants import PROJ_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def train_logreg(X: np.ndarray, y: List[int]) -> LogisticRegression:
    """Train a logistic regression model on the given data."""
    clf = LogisticRegression(random_state=0, multi_class="multinomial").fit(X, y)
    return clf


def main():
    """Train models."""
    # read in data
    print("Reading in data .....")
    subdir = "lim1000_data"
    full_fp = os.path.join(PROJ_DIR, subdir, "full_data.csv")
    df = pd.read_csv(full_fp, header=None)
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train models
    print("Training models .....")
    logreg = train_logreg(X_train, y_train)

    # evaluate models
    print("Evaluating models .....")
    log_preds = logreg.predict(X_test)
    acc = accuracy_score(y_test, log_preds)
    rec = recall_score(y_test, log_preds, average='weighted')
    prec = recall_score(y_test, log_preds, average='weighted')

    # return evals
    eval_df = pd.DataFrame({"metric": ["accuracy", "recall", "precision"],
                            "result": [acc, rec, prec]})
    eval_df.to_csv(os.path.join(PROJ_DIR, subdir, "evals.csv"), index=False)


if __name__ == "__main__":
    main()
