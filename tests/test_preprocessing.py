"""Unit tests for the preprocessing pipeline (run with ``pytest -q``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    expand_contractions,
    mark_negation,
    normalize,
    preprocess,
    remove_html,
    remove_noise,
)


def test_remove_html():
    cleaned = remove_html("<br />Great movie &amp; story")
    assert cleaned.strip() == "Great movie & story"
    assert "<" not in remove_html("<p>Hello</p>")


def test_normalize_removes_accents_and_lowercases():
    assert normalize("Café ÉPICO") == "cafe epico"


def test_expand_contractions_creates_explicit_negation():
    assert "not" in expand_contractions("i don't like it").split()
    assert expand_contractions("can't") == "cannot"


def test_remove_noise_drops_urls_numbers_punctuation():
    cleaned = remove_noise("visit http://x.com now, 10/10!")
    assert "http" not in cleaned and "10" not in cleaned and "," not in cleaned


def test_negation_does_not_cross_clause_boundary():
    assert mark_negation(["not", "good", "|", "cast"]) == ["not", "not_good", "cast"]


def test_mark_negation_marks_three_following_tokens():
    assert mark_negation(["not", "good", "at", "all", "though"]) == [
        "not", "not_good", "not_at", "not_all", "though"
    ]


def test_mark_negation_resets_scope():
    assert mark_negation(["no", "doubt", "not", "bad"]) == [
        "no", "not_doubt", "not", "not_bad"
    ]


def test_preprocess_keeps_negation_information():
    tokens = preprocess("This movie is not good at all.").split()
    assert any(token.startswith("not_") for token in tokens)
    assert "not" in tokens


def test_preprocess_is_deterministic():
    text = "A <b>great</b> film, isn't it?"
    assert preprocess(text) == preprocess(text)


def test_preprocess_removes_stopwords():
    assert "the" not in preprocess("The film was the best.").split()
