#!/usr/bin/env python
"""Command-line entry point that reproduces every number reported in the paper.

Examples
--------
Smoke test on a small stratified sample (2 000 reviews per class)::

    python scripts/run_experiments.py --sample 2000 --results-dir results/smoke

Full experiment (50 000 reviews, four classifiers, 5-fold CV)::

    python scripts/run_experiments.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_ORDER, RESULTS_DIR, SEED  # noqa: E402
from src.data import load_dataset, split_dataset  # noqa: E402
from src.preprocessing import ensure_nltk_data  # noqa: E402

LOGGER = logging.getLogger("sentiment")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", *MODEL_ORDER],
                        help="models to run (default: all)")
    parser.add_argument("--sample", type=int, default=None,
                        help="reviews per class, for a quick smoke test")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the joblib cache of the TF-IDF/SVD steps")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="directory where the result tables are written")
    return parser.parse_args(argv)


def setup_logging(results_dir: Path) -> None:
    """Send every module logger to the console and to ``results/run.log``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(results_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        root.addHandler(handler)


def corpus_statistics(frame: pd.DataFrame) -> dict:
    """Descriptive statistics of the corpus, reported in Table 1 of the paper."""
    raw_words = frame["text"].str.split().str.len()
    clean_tokens = frame["clean"].str.split().str.len()
    stats = {
        "n_documents": int(len(frame)),
        "n_positive": int((frame["label"] == 1).sum()),
        "n_negative": int((frame["label"] == 0).sum()),
        "raw_words_mean": float(raw_words.mean()),
        "raw_words_median": float(raw_words.median()),
        "raw_words_std": float(raw_words.std(ddof=1)),
        "raw_words_max": int(raw_words.max()),
        "tokens_mean": float(clean_tokens.mean()),
        "tokens_median": float(clean_tokens.median()),
        "tokens_std": float(clean_tokens.std(ddof=1)),
        "tokens_max": int(clean_tokens.max()),
    }
    vocabulary = set()
    for document in frame["clean"]:
        vocabulary.update(document.split())
    stats["vocabulary_size"] = len(vocabulary)
    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.results_dir)
    ensure_nltk_data()

    np.random.seed(SEED)

    # ---- data ------------------------------------------------------------- #
    frame = load_dataset(use_cache=True, sample=args.sample)
    stats = corpus_statistics(frame)
    with (args.results_dir / "corpus_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Corpus: %d documents | %.1f raw words/review | %.1f tokens after cleaning",
                stats["n_documents"], stats["raw_words_mean"], stats["tokens_mean"])

    X_train, X_test, y_train, y_test = split_dataset(frame)
    LOGGER.info("Vocabulary (whole corpus, after preprocessing): %d types",
                stats["vocabulary_size"])

    # ---- experiment -------------------------------------------------------- #
    from src.experiment import run_experiment

    models = list(MODEL_ORDER) if args.models == ["all"] else args.models
    LOGGER.info("Running models: %s", ", ".join(models))

    run_experiment(
        X_train, y_train, X_test, y_test,
        models=models,
        cache=not args.no_cache,
        results_dir=args.results_dir,
    )

    # ---- console summary --------------------------------------------------- #
    table = pd.read_csv(args.results_dir / "metrics_test.csv")
    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    LOGGER.info("Test-set results:\n%s", table[columns].to_string(index=False, float_format="%.4f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
