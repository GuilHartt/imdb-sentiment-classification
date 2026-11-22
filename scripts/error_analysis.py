#!/usr/bin/env python
"""Error analysis of the trained classifiers on the held-out test set.

Produces ``results/error_analysis.json`` (rates) and ``results/error_examples.csv``
(misclassified reviews) so that the discussion in the paper is grounded on the
predictions actually produced by the models.

Usage::

    python scripts/error_analysis.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_dataset, split_dataset  # noqa: E402

NEGATION_CUES = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nowhere)\b|n't", re.I)
CONTRAST_CUES = re.compile(
    r"\b(?:but|however|although|though|yet|nevertheless|despite|nonetheless)\b", re.I
)
IRONY_CUES = re.compile(r"\b(?:oh great|yeah right|just what i needed|what a surprise|"
                        r"thanks a lot|as if)\b", re.I)


def main() -> int:
    frame = load_dataset(use_cache=True)
    _, X_test, _, y_test = split_dataset(frame)

    # Recover the raw texts aligned with the saved predictions (same split).
    from sklearn.model_selection import train_test_split
    from src.config import SEED, TEST_SIZE

    _, raw_test, _, _ = train_test_split(
        frame["text"], frame["label"], test_size=TEST_SIZE, stratify=frame["label"],
        random_state=SEED,
    )
    raw_test = raw_test.reset_index(drop=True)
    y_true = np.asarray(y_test)

    summary: dict[str, dict] = {}
    examples: list[dict] = []

    for path in sorted(PREDICTIONS_DIR.glob("*.npz")):
        model = path.stem
        data = np.load(path)
        assert np.array_equal(data["y_true"], y_true), f"split mismatch for {model}"
        y_pred = data["y_pred"]
        errors = y_pred != y_true

        lengths = raw_test.str.split().str.len().to_numpy()
        quartiles = pd.qcut(lengths, 4, labels=["Q1", "Q2", "Q3", "Q4"])

        negations = raw_test.str.contains(NEGATION_CUES).to_numpy()
        contrasts = raw_test.str.contains(CONTRAST_CUES).to_numpy()
        irony = raw_test.str.contains(IRONY_CUES).to_numpy()

        summary[model] = {
            "n_errors": int(errors.sum()),
            "error_rate": float(errors.mean()),
            "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
            "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum()),
            "error_rate_with_negation": float(errors[negations].mean()),
            "error_rate_without_negation": float(errors[~negations].mean()),
            "share_with_negation": float(negations.mean()),
            "error_rate_with_contrast": float(errors[contrasts].mean()),
            "error_rate_without_contrast": float(errors[~contrasts].mean()),
            "error_rate_with_irony_cue": float(errors[irony].mean()) if irony.any() else None,
            "n_irony_cue": int(irony.sum()),
            "mean_length_error": float(lengths[errors].mean()),
            "mean_length_correct": float(lengths[~errors].mean()),
            "error_rate_by_length_quartile": {
                str(label): float(errors[quartiles == label].mean())
                for label in ["Q1", "Q2", "Q3", "Q4"]
            },
        }

    # Examples of the best model (highest test F1).
    metrics = pd.read_csv(RESULTS_DIR / "metrics_test.csv").set_index("model")
    best_model = metrics["f1"].idxmax()
    data = np.load(PREDICTIONS_DIR / f"{best_model}.npz")
    y_pred = data["y_pred"]
    errors = y_pred != y_true

    def excerpt(text: str, n: int = 45) -> str:
        words = text.split()
        return " ".join(words[:n])

    false_positives = np.where(errors & (y_pred == 1))[0][:4]
    false_negatives = np.where(errors & (y_pred == 0))[0][:4]
    for index in false_positives:
        examples.append({"model": best_model, "type": "falso positivo", "index": int(index),
                         "true_label": int(y_true[index]), "pred": int(y_pred[index]),
                         "text": excerpt(raw_test.iloc[index])})
    for index in false_negatives:
        examples.append({"model": best_model, "type": "falso negativo", "index": int(index),
                         "true_label": int(y_true[index]), "pred": int(y_pred[index]),
                         "text": excerpt(raw_test.iloc[index])})

    with (RESULTS_DIR / "error_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_model": best_model, "models": summary}, handle, indent=2,
                  ensure_ascii=False)
    pd.DataFrame(examples).to_csv(RESULTS_DIR / "error_examples.csv", index=False,
                                  quoting=csv.QUOTE_ALL)

    print(json.dumps({"best_model": best_model, "models": summary}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
