"""Estimators and TF-IDF configuration used in the benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import SEED, SVD_COMPONENTS, TFIDF_PARAMS


def make_tfidf() -> TfidfVectorizer:
    """TF-IDF vectoriser with the parameters fixed for every experiment."""
    params = dict(TFIDF_PARAMS)
    params["dtype"] = np.float32
    return TfidfVectorizer(**params)


def make_pipeline(model_name: str, seed: int = SEED) -> Pipeline:
    """Return an unfitted pipeline ``tfidf [-> svd -> scaler] -> classifier``."""
    steps = [("tfidf", make_tfidf())]

    if model_name == "nb":
        classifier = MultinomialNB()
    elif model_name == "logreg":
        classifier = LogisticRegression(
            solver="liblinear", max_iter=2000, random_state=seed
        )
    elif model_name == "svm":
        classifier = LinearSVC(dual=True, max_iter=5000, random_state=seed)
    elif model_name == "mlp":
        steps += [
            ("svd", TruncatedSVD(
                n_components=SVD_COMPONENTS, algorithm="randomized",
                n_iter=5, random_state=seed,
            )),
            ("scaler", StandardScaler()),
        ]
        classifier = MLPClassifier(
            solver="adam", batch_size=256, max_iter=60, tol=1e-4,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    steps.append(("clf", classifier))
    return Pipeline(steps)


def decision_scores(estimator: Pipeline, X) -> np.ndarray:
    """Continuous score of the positive class (probability or margin)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)
