"""Central configuration of the IMDb sentiment-classification benchmark.

Every value that influences an experimental result lives here, so that a single
file documents the protocol and any change is explicit in the version history.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_ARCHIVE = RAW_DIR / "aclImdb_v1.tar.gz"
IMDB_DIR = RAW_DIR / "aclImdb"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.20          # 80 % train / 20 % test, stratified
N_SPLITS = 5              # folds used for hyper-parameter search and CV estimate

# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
#: Words that invert the polarity of the tokens that follow them.
NEGATION_WORDS = frozenset({"not", "no", "never"})
#: Number of tokens after a negation word that receive the ``not_`` prefix.
NEGATION_SCOPE = 3

# --------------------------------------------------------------------------- #
# Vectorisation (TF-IDF)
# --------------------------------------------------------------------------- #
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 50_000,
    "sublinear_tf": True,
    "lowercase": False,          # text is already lower-cased in preprocessing
    "token_pattern": r"(?u)\S+",  # tokens are pre-segmented by NLTK
    "dtype": "float32",
}

#: Dimensionality reduction applied before the MLP (dense, low dimensional input).
SVD_COMPONENTS = 300

# --------------------------------------------------------------------------- #
# Model grids
# --------------------------------------------------------------------------- #
#: Hyper-parameter grids explored with exhaustive grid search (GridSearchCV).
PARAM_GRIDS: dict[str, dict] = {
    "nb": {"clf__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
    "logreg": {"clf__C": [0.1, 1.0, 10.0, 100.0]},
    "svm": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "mlp": {
        "clf__hidden_layer_sizes": [(100,), (300,)],
        "clf__alpha": [1e-4, 1e-2],
    },
}

MODEL_LABELS: dict[str, str] = {
    "nb": "Naive Bayes Multinomial",
    "logreg": "Regressão Logística",
    "svm": "SVM Linear",
    "mlp": "MLP",
}

MODEL_ORDER = ["nb", "svm", "logreg", "mlp"]

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
N_BOOTSTRAP = 1000        # resamples for the bootstrap confidence intervals
