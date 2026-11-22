"""Experimental protocol: grid search, cross-validation, test evaluation, timing."""

from __future__ import annotations

import json
import logging
import pickle
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Memory
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import (
    CACHE_DIR,
    MODEL_LABELS,
    MODEL_ORDER,
    N_SPLITS,
    PARAM_GRIDS,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SEED,
)
from .evaluation import bootstrap_f1_ci, classification_metrics, mcnemar_test
from .models import make_pipeline

LOGGER = logging.getLogger(__name__)

#: Every metric is computed inside the cross-validation loop, so that the
#: reported mean and standard deviation come from the very same partitions
#: used to select the hyper-parameters.
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
METRICS = list(SCORING)


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    """Everything the paper needs to report about one classifier."""

    model: str
    best_params: dict
    fold_metrics: pd.DataFrame          # one row per CV fold, all metrics
    test_metrics: dict
    f1_ci: tuple[float, float]
    timing: dict[str, float]
    model_size_mb: float
    n_parameters: int
    vocabulary_size: int
    grid: pd.DataFrame = field(repr=False)
    estimator: object = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)

    def cv_summary(self) -> dict:
        """Mean and standard deviation of every metric across the CV folds."""
        summary = {}
        for column in METRICS:
            summary[f"{column}_mean"] = float(self.fold_metrics[column].mean())
            summary[f"{column}_std"] = float(self.fold_metrics[column].std(ddof=1))
        return summary


# --------------------------------------------------------------------------- #
# Environment / persistence helpers
# --------------------------------------------------------------------------- #
def environment_info() -> dict:
    """Versions and machine description that make the run reproducible."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "seed": SEED,
        "n_splits": N_SPLITS,
        "scoring": SCORING,
    }


def _count_parameters(estimator) -> int:
    """Number of trainable parameters of the classifier (weights + biases)."""
    clf = estimator.named_steps["clf"]
    total = 0
    for attribute in ("coef_", "intercept_", "feature_log_prob_"):
        value = getattr(clf, attribute, None)
        if value is not None:
            total += int(np.size(value))
    if hasattr(clf, "coefs_"):  # MLPClassifier
        total += sum(int(weight.size) for weight in clf.coefs_)
        total += sum(int(bias.size) for bias in clf.intercepts_)
    return total


def _model_size_mb(estimator) -> float:
    """Serialised size of the *whole* pipeline (vectoriser + classifier), in MB."""
    return round(len(pickle.dumps(estimator, protocol=5)) / (1024 ** 2), 3)


# --------------------------------------------------------------------------- #
# One model
# --------------------------------------------------------------------------- #
def run_model(
    model_name: str,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    cache: bool = True,
) -> ModelResult:
    """Grid search with 5-fold CV, then a single evaluation on the test set.

    The hyper-parameters are selected by exhaustive grid search (GridSearchCV)
    maximising the F1-score, using the same five stratified folds that produce
    the cross-validation estimate.  The selected configuration is then refitted
    on the whole training set and evaluated once on the held-out test set.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("Model: %s (%s)", model_name, MODEL_LABELS[model_name])
    LOGGER.info("=" * 70)

    memory = Memory(location=CACHE_DIR / "sklearn", verbose=0) if cache else None
    pipeline = make_pipeline(model_name)
    pipeline.memory = memory

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS[model_name],
        scoring=SCORING,
        refit="f1",
        cv=cv,
        n_jobs=1,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    search_time = time.perf_counter() - start

    best_index = search.best_index_
    fold_metrics = pd.DataFrame({
        metric: [search.cv_results_[f"split{k}_test_{metric}"][best_index]
                 for k in range(N_SPLITS)]
        for metric in METRICS
    })
    fold_metrics.insert(0, "fold", np.arange(1, N_SPLITS + 1))
    fold_metrics["fit_time"] = search.cv_results_["mean_fit_time"][best_index]

    LOGGER.info("Grid search: %.1f s | best F1 (CV) = %.4f | %s",
                search_time, search.best_score_, search.best_params_)
    for row in fold_metrics.itertuples():
        LOGGER.info("  fold %d: F1=%.4f acc=%.4f recall=%.4f",
                    row.fold, row.f1, row.accuracy, row.recall)

    # ---- cost of fitting the full pipeline on the whole training set -------- #
    # The refit is performed without the joblib cache, so the measured time is
    # the real cost of vectorising 40 000 documents and training the classifier.
    final_model = make_pipeline(model_name)
    transform_start = time.perf_counter()
    X_train_transformed = final_model[:-1].fit_transform(X_train)
    transform_time = time.perf_counter() - transform_start

    fit_start = time.perf_counter()
    final_model[-1].fit(X_train_transformed, y_train)
    classifier_fit_time = time.perf_counter() - fit_start
    del X_train_transformed

    predict_start = time.perf_counter()
    X_test_transformed = final_model[:-1].transform(X_test)
    transform_predict_time = time.perf_counter() - predict_start
    predict_start = time.perf_counter()
    y_pred_test = final_model[-1].predict(X_test_transformed)
    classifier_predict_time = time.perf_counter() - predict_start

    if hasattr(final_model[-1], "predict_proba"):
        y_score_test = final_model[-1].predict_proba(X_test_transformed)[:, 1]
    else:
        y_score_test = final_model[-1].decision_function(X_test_transformed)

    test_metrics = classification_metrics(y_test, y_pred_test, y_score_test)
    f1_ci = bootstrap_f1_ci(y_test, y_pred_test)
    LOGGER.info("Test: F1=%.4f acc=%.4f AUC=%.4f (IC 95%% F1: %.4f-%.4f)",
                test_metrics["f1"], test_metrics["accuracy"], test_metrics["roc_auc"],
                f1_ci[0], f1_ci[1])

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREDICTIONS_DIR / f"{model_name}.npz",
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred_test),
        y_score=np.asarray(y_score_test),
    )

    total_fit_time = transform_time + classifier_fit_time
    timing = {
        "grid_search_s": round(search_time, 2),
        "preprocessing_s": round(transform_time, 3),
        "classifier_fit_s": round(classifier_fit_time, 3),
        "final_fit_s": round(total_fit_time, 3),
        "predict_transform_s": round(transform_predict_time, 3),
        "predict_s": round(classifier_predict_time, 3),
        "predict_total_ms": round(1000 * (transform_predict_time + classifier_predict_time), 1),
        "predict_per_doc_us": round(
            1e6 * (transform_predict_time + classifier_predict_time) / len(X_test), 2
        ),
    }

    size = _model_size_mb(final_model)
    LOGGER.info("Cost: vectorisation %.1f s + classifier %.1f s | %.3f MB | %d params",
                transform_time, classifier_fit_time, size, _count_parameters(final_model))

    return ModelResult(
        model=model_name,
        best_params=search.best_params_,
        fold_metrics=fold_metrics,
        test_metrics=test_metrics,
        f1_ci=f1_ci,
        timing=timing,
        model_size_mb=size,
        n_parameters=_count_parameters(final_model),
        vocabulary_size=len(final_model.named_steps["tfidf"].vocabulary_),
        grid=pd.DataFrame(search.cv_results_),
        estimator=final_model,
    )


def top_features(estimator, n: int = 20) -> pd.DataFrame:
    """Most positive and most negative TF-IDF weights of a linear classifier."""
    vocabulary = estimator.named_steps["tfidf"].get_feature_names_out()
    coefficients = estimator.named_steps["clf"].coef_.ravel()
    order = np.argsort(coefficients)
    return pd.concat([
        pd.DataFrame({"term": vocabulary[order[:n]], "weight": coefficients[order[:n]],
                      "polarity": "negative"}),
        pd.DataFrame({"term": vocabulary[order[::-1][:n]],
                      "weight": coefficients[order[::-1][:n]],
                      "polarity": "positive"}),
    ], ignore_index=True)


# --------------------------------------------------------------------------- #
# Full experiment
# --------------------------------------------------------------------------- #
def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    models: list[str] | None = None,
    cache: bool = True,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, ModelResult]:
    """Run every requested model and write the result tables to ``results_dir``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    models = models or MODEL_ORDER
    results: dict[str, ModelResult] = {}

    for model_name in models:
        results[model_name] = run_model(
            model_name, X_train, y_train, X_test, y_test, cache=cache
        )

    # ---- long/wide tables -------------------------------------------------- #
    cv_table = pd.concat(
        [result.fold_metrics.assign(model=name) for name, result in results.items()],
        ignore_index=True,
    )
    cv_table.to_csv(results_dir / "metrics_cv.csv", index=False)

    test_rows = []
    for name, result in results.items():
        row = {"model": name, "label": result.label, **result.test_metrics}
        row.update(result.cv_summary())
        row.update(result.timing)
        row.update({
            "f1_ci_low": result.f1_ci[0],
            "f1_ci_high": result.f1_ci[1],
            "model_size_mb": result.model_size_mb,
            "n_parameters": result.n_parameters,
            "vocabulary_size": result.vocabulary_size,
            "best_params": json.dumps(result.best_params, ensure_ascii=False),
        })
        test_rows.append(row)
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(results_dir / "metrics_test.csv", index=False)

    # ---- hyper-parameter search tables ------------------------------------ #
    with (results_dir / "grid_search.csv").open("w", encoding="utf-8") as handle:
        for name, result in results.items():
            grid = result.grid
            columns = ["params", "mean_test_f1", "std_test_f1", "mean_fit_time"] + [
                column for column in grid.columns
                if column.startswith("mean_test_") and column != "mean_test_f1"
            ]
            block = grid[columns].copy()
            block.insert(0, "model", name)
            block.to_csv(handle, index=False)
            handle.write("\n")

    # ---- most discriminative terms of the linear models -------------------- #
    for name, result in results.items():
        if result.estimator is not None and hasattr(result.estimator.named_steps["clf"], "coef_"):
            top_features(result.estimator).to_csv(
                results_dir / f"top_features_{name}.csv", index=False
            )

    # ---- paired McNemar tests between all pairs --------------------------- #
    names = list(results)
    rows = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            preds_a = np.load(PREDICTIONS_DIR / f"{name_a}.npz")
            preds_b = np.load(PREDICTIONS_DIR / f"{name_b}.npz")
            stats = mcnemar_test(preds_a["y_true"], preds_a["y_pred"], preds_b["y_pred"])
            rows.append({"model_a": name_a, "model_b": name_b, **stats})
    pd.DataFrame(rows).to_csv(results_dir / "mcnemar.csv", index=False)

    # ---- run metadata ------------------------------------------------------ #
    metadata = {
        "environment": environment_info(),
        "models": list(results),
        "best_params": {name: r.best_params for name, r in results.items()},
        "timing": {name: r.timing for name, r in results.items()},
        "vocabulary_size": {name: r.vocabulary_size for name, r in results.items()},
    }
    with (results_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, default=str)

    LOGGER.info("Results written to %s", results_dir)
    return results
