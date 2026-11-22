"""Text preprocessing for the IMDb sentiment-classification benchmark.

The pipeline is deterministic and independent of the corpus statistics: no
quantity is estimated from the data, therefore applying it before the
train/test split does not leak information.  The exact sequence of operations
is described in Section "Materiais e Métodos" of the accompanying paper.
"""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import NEGATION_SCOPE, NEGATION_WORDS

# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
_STEMMER = PorterStemmer()

#: sklearn's English stop-word list, without the negation words: removing them
#: would destroy the polarity information we explicitly want to model.
STOPWORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) - NEGATION_WORDS

#: Common English contractions.  ``n't`` is handled separately so that any
#: contracted negation becomes the explicit token *not*.
CONTRACTIONS: dict[str, str] = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "isn't": "is not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
    "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "you'd": "you would", "he's": "he is", "he'd": "he would",
    "he'll": "he will", "she's": "she is", "she'd": "she would",
    "she'll": "she will", "it's": "it is", "it'd": "it would",
    "it'll": "it will", "we're": "we are", "we've": "we have",
    "we'll": "we will", "we'd": "we would", "they're": "they are",
    "they've": "they have", "they'll": "they will", "they'd": "they would",
    "that's": "that is", "that'd": "that would", "that'll": "that will",
    "there's": "there is", "there'd": "there would", "there'll": "there will",
    "what's": "what is", "what're": "what are", "what'll": "what will",
    "who's": "who is", "who'll": "who will", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
    "let's": "let us", "here's": "here is",
    "y'all": "you all", "ma'am": "madam", "o'clock": "of the clock",
}

_CONTRACTION_RE = re.compile(
    "|".join(re.escape(key) for key in sorted(CONTRACTIONS, key=len, reverse=True))
)
_NEG_CONTRACTION_RE = re.compile(r"n['’]t\b")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:\S+\.(?:com|org|net|edu|gov|br|co|uk)\b)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")
_SPACES_RE = re.compile(r"\s+")

#: Marker inserted where the original text had punctuation.  It is *not* a
#: feature: it only prevents a negation from acting beyond a clause boundary
#: ("not a good film, but the cast is great" must not mark "cast").
CLAUSE_MARKER = "|"


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def remove_html(text: str) -> str:
    """Strip HTML tags and decode HTML entities (``&amp;`` -> ``&``)."""
    return html.unescape(_HTML_TAG_RE.sub(" ", text))


def normalize(text: str) -> str:
    """Lower-case and remove diacritics (``café`` -> ``cafe``)."""
    text = text.lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def expand_contractions(text: str) -> str:
    """Expand English contractions so that negations become explicit tokens."""
    text = _CONTRACTION_RE.sub(lambda match: CONTRACTIONS[match.group(0)], text)
    return _NEG_CONTRACTION_RE.sub(" not", text)


def remove_noise(text: str) -> str:
    """Remove URLs, numbers and punctuation, collapsing redundant spaces.

    Punctuation is replaced by :data:`CLAUSE_MARKER` instead of a space so that
    clause boundaries remain available to the negation marker.
    """
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(f" {CLAUSE_MARKER} ", text)
    return _SPACES_RE.sub(" ", text).strip()


@lru_cache(maxsize=250_000)
def stem(token: str) -> str:
    """Porter stem of a single token, memoised to keep the cost linear."""
    return _STEMMER.stem(token)


def mark_negation(tokens: list[str]) -> list[str]:
    """Prefix the tokens inside a negation scope with ``not_``.

    ``["not", "good", "at", "all"]`` becomes ``["not", "not_good", "not_at",
    "not_all"]``, so that a linear model can learn a weight for the negated
    form instead of reusing the positive one (PANG; LEE; VAITHYANATHAN, 2002).
    The scope stops after :data:`NEGATION_SCOPE` tokens or at a clause
    boundary, whichever comes first.
    """
    marked: list[str] = []
    remaining = 0
    for token in tokens:
        if token == CLAUSE_MARKER:
            remaining = 0
            continue
        if token in NEGATION_WORDS:
            marked.append(token)
            remaining = NEGATION_SCOPE
        elif remaining > 0:
            marked.append(f"not_{token}")
            remaining -= 1
        else:
            marked.append(token)
    return marked


def tokenize(text: str) -> list[str]:
    """Tokenise with NLTK and drop stop words (negations and markers are kept)."""
    return [
        token for token in word_tokenize(text)
        if (token == CLAUSE_MARKER)
        or (token not in STOPWORDS and len(token) > 1)
    ]


def preprocess(text: str) -> str:
    """Full pipeline: raw review -> space-separated stemmed token sequence."""
    text = remove_html(text)
    text = normalize(text)
    text = expand_contractions(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = mark_negation(tokens)
    return " ".join(
        f"not_{stem(token[4:])}" if token.startswith("not_") else stem(token)
        for token in tokens
    )


def ensure_nltk_data() -> None:
    """Download the NLTK tokeniser models required by :func:`word_tokenize`."""
    import nltk

    for package in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            nltk.download(package, quiet=True)


# Marks the version of the pipeline; used as the cache key of preprocessed data.
PREPROCESSING_VERSION = "v1"
