#!/usr/bin/env python
"""Analyse how the negation marking is reflected in the learned weights.

The paper reports (Section 5.5) how many features carry the ``not_`` prefix,
how much of the total absolute weight they concentrate and which negated forms
received the largest positive/negative weights.  This script reproduces those
numbers from a model refitted with the selected hyper-parameters.

Usage::

    python scripts/negation_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR  # noqa: E402
from src.data import load_dataset, split_dataset  # noqa: E402
from src.models import make_pipeline  # noqa: E402

NEGATION_PREFIX = "not_"
SVM_C = 0.1  # value selected by the grid search on the 5-fold cross-validation


def main() -> int:
    frame = load_dataset(use_cache=True)
    X_train, _, y_train, _ = split_dataset(frame)

    pipeline = make_pipeline("svm")
    pipeline.named_steps["clf"].set_params(C=SVM_C)
    pipeline.fit(X_train, y_train)

    vocabulary = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    coefficients = pipeline.named_steps["clf"].coef_.ravel()

    mask = np.char.startswith(vocabulary.astype(str), NEGATION_PREFIX)
    negated, weights = vocabulary[mask], coefficients[mask]
    order = np.argsort(weights)

    summary = {
        "n_features": int(len(vocabulary)),
        "n_negated_features": int(mask.sum()),
        "share_of_absolute_weight": float(np.abs(weights).sum() / np.abs(coefficients).sum()),
        "most_positive": {str(negated[i]): float(weights[i]) for i in order[::-1][:5]},
        "most_negative": {str(negated[i]): float(weights[i]) for i in order[:5]},
    }

    (RESULTS_DIR / "negation_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame({
        "term": negated[order[::-1][:20]],
        "weight": weights[order[::-1][:20]],
    }).to_csv(RESULTS_DIR / "top_negation_features.csv", index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
