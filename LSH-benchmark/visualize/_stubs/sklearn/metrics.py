from __future__ import annotations

from typing import Iterable


def _counts(y_true: Iterable[int], y_pred: Iterable[int]) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        yt = int(yt)
        yp = int(yp)
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_score(y_true, y_pred):
    tp, fp, _, _ = _counts(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall_score(y_true, y_pred):
    tp, _, fn, _ = _counts(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def balanced_accuracy_score(y_true, y_pred):
    tp, fp, fn, tn = _counts(y_true, y_pred)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


def roc_auc_score(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

