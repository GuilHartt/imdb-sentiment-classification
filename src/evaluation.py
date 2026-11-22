"""Metrics, confidence intervals and paired statistical tests."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import N_BOOTSTRAP, SEED

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC AUC plus the confusion counts."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_f1_ci(
    y_true, y_pred, n_boot: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the F1-score of the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        values[i] = f1_score(y_true[idx], y_pred[idx], zero_division=0)
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict[str, float]:
    """McNemar test for two classifiers evaluated on the same test set.

    Compares the discordant predictions only.  The exact binomial test is used
    when there are fewer than 25 discordant pairs, otherwise the chi-square
    statistic with continuity correction (as recommended for this test).
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true
    correct_b = np.asarray(y_pred_b) == y_true
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    if n == 0:
        statistic, p_value = 0.0, 1.0
    elif n < 25:
        statistic = float(b)
        p_value = float(stats.binomtest(b, n, 0.5).pvalue)
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n)
        p_value = float(stats.chi2.sf(statistic, df=1))
    return {"b": b, "c": c, "statistic": statistic, "p_value": p_value}


def wilcoxon_paired(scores_a, scores_b) -> dict[str, float]:
    """Wilcoxon signed-rank test between paired fold-level scores."""
    statistic, p_value = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox")
    return {"statistic": float(statistic), "p_value": float(p_value)}
