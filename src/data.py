"""Dataset acquisition and loading for the IMDb benchmark."""

from __future__ import annotations

import logging
import pickle
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CACHE_DIR, IMDB_ARCHIVE, IMDB_DIR, IMDB_URL, RAW_DIR, SEED, TEST_SIZE
from .preprocessing import PREPROCESSING_VERSION, preprocess

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False) -> Path:
    """Download the ``aclImdb`` archive (84 MB) if it is not present yet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if IMDB_ARCHIVE.exists() and not force:
        LOGGER.info("Archive already present: %s", IMDB_ARCHIVE)
        return IMDB_ARCHIVE

    LOGGER.info("Downloading %s -> %s", IMDB_URL, IMDB_ARCHIVE)
    with urllib.request.urlopen(IMDB_URL) as response, IMDB_ARCHIVE.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                LOGGER.info("  %.1f%% (%d/%d MB)", 100 * done / total, done >> 20, total >> 20)
    return IMDB_ARCHIVE


def extract_dataset(force: bool = False) -> Path:
    """Extract the archive; the resulting root is ``data/raw/aclImdb``."""
    if (IMDB_DIR / "train" / "pos").is_dir() and not force:
        LOGGER.info("Dataset already extracted: %s", IMDB_DIR)
        return IMDB_DIR
    archive = download_dataset()
    LOGGER.info("Extracting %s", archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(RAW_DIR, filter="data")
    return IMDB_DIR


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_split(root: Path, split: str) -> list[tuple[str, int, str]]:
    """Read ``pos``/``neg`` folders of one split; label 1 = positive."""
    records: list[tuple[str, int, str]] = []
    for label_name, label in (("pos", 1), ("neg", 0)):
        folder = root / split / label_name
        for path in sorted(folder.glob("*.txt")):
            records.append((path.read_text(encoding="utf-8", errors="replace"), label, split))
    return records


def build_dataframe() -> pd.DataFrame:
    """Read the 50 000 reviews into a ``text``/``label``/``original_split`` frame."""
    root = extract_dataset()
    records = _read_split(root, "train") + _read_split(root, "test")
    frame = pd.DataFrame(records, columns=["text", "label", "original_split"])
    frame = frame.sort_values(["original_split", "label"], kind="stable").reset_index(drop=True)
    LOGGER.info("Loaded %d reviews (%d positive / %d negative)",
                len(frame), int(frame["label"].sum()), int((1 - frame["label"]).sum()))
    return frame


def load_dataset(use_cache: bool = True, sample: int | None = None) -> pd.DataFrame:
    """Return the corpus with an extra ``clean`` column holding processed text.

    Parameters
    ----------
    use_cache:
        Reuse ``data/cache/imdb_preprocessed_<version>[_<sample>].pkl`` when
        available; preprocessing 50 000 reviews takes a few minutes.
    sample:
        Keep only the first ``sample`` reviews per class (smoke tests).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{PREPROCESSING_VERSION}"
    if sample:
        suffix += f"_n{sample}"
    cache_file = CACHE_DIR / f"imdb_preprocessed_{suffix}.pkl"

    if use_cache and cache_file.exists():
        LOGGER.info("Loading cached preprocessed corpus: %s", cache_file)
        return pd.read_pickle(cache_file)

    frame = build_dataframe()
    if sample:
        frame = (
            frame.groupby("label", group_keys=False)
            .head(sample)
            .reset_index(drop=True)
        )
        LOGGER.info("Subsampled to %d reviews (%d per class)", len(frame), sample)

    LOGGER.info("Preprocessing %d reviews (this may take a few minutes)...", len(frame))
    cleaned: list[str] = []
    step = max(len(frame) // 10, 1)
    for i, text in enumerate(frame["text"]):
        cleaned.append(preprocess(text))
        if (i + 1) % step == 0:
            LOGGER.info("  processed %d/%d", i + 1, len(frame))
    frame["clean"] = cleaned
    frame = frame[["text", "clean", "label", "original_split"]]

    frame.to_pickle(cache_file)
    LOGGER.info("Cached preprocessed corpus: %s", cache_file)
    return frame


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified 80/20 split; returns ``X_train, X_test, y_train, y_test``."""
    X = frame["clean"]
    y = frame["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    LOGGER.info(
        "Split: %d train (%.1f%% positive) / %d test (%.1f%% positive)",
        len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def save_text_corpus(frame: pd.DataFrame, path: Path) -> None:
    """Persist the corpus as ``pickle`` (keeps the cleaning identical across runs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
