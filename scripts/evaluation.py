"""Define functions used for evaluation."""
from typing import List, Set, Tuple
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd


def probs_to_preds(probs, threshold):
    """Classifies probabilities with the theshold."""
    return [[0 if p < threshold else 1 for p in row]
            for row in probs]


def ml_accuracy(Y_true: List[List[int]], Y_pred: List[List[int]]) -> float:
    """Calculate multi-label accuracy."""
    ratios = []
    for z, y in zip(Y_true, Y_pred):
        iz = [i for i, val in enumerate(z) if val]
        iy = [i for i, val in enumerate(y) if val]
        sz, sy = set(iz), set(iy)
        ratio = len(sz & sy) / len(sz | sy)
        ratios.append(ratio)
    acc = sum(ratios) / len(ratios)
    return acc


def pr_curve(clf_name, probs, y_true, thresholds):
    """
    Generate precision/recall curve data for a given classifier.

    Implementation is custom because sklearn doesn't
    support multilabel classification for pr curve.
    """
    precs = []
    recs = []
    for thresh in tqdm(thresholds):
        preds = probs_to_preds(probs, thresh)
        precs.append(precision_score(y_true, preds, average="weighted", zero_division=1))
        recs.append(recall_score(y_true, preds, average="weighted", zero_division=1))
    data = {"Classifier": [clf_name] * len(thresholds),
            "Precision": precs,
            "Recall": recs,
            "Threshold": thresholds}
    return pd.DataFrame(data)
