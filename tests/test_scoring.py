"""Tests for scoring and normalization."""

from stt_bench.scoring.normalize import normalize
from stt_bench.scoring.score import aggregate_scores, score_sample


def test_normalize_lowercase():
    assert normalize("Hello World") == "hello world"


def test_normalize_punctuation():
    assert normalize("Hello, world!") == "hello world"


def test_normalize_keep_apostrophe():
    assert normalize("don't stop") == "don't stop"


def test_normalize_whitespace():
    assert normalize("  hello   world  ") == "hello world"


def test_normalize_unicode():
    assert normalize("café résumé") == "café résumé"


def test_score_perfect_match():
    score = score_sample("v1", "m1", "hello world", "hello world")
    assert score.wer == 0.0
    assert score.cer == 0.0
    assert score.insertions == 0
    assert score.deletions == 0
    assert score.substitutions == 0


def test_score_one_substitution():
    score = score_sample("v1", "m1", "hello world", "hello earth")
    assert score.wer == 0.5  # 1 of 2 words wrong
    assert score.substitutions == 1


def test_score_one_insertion():
    score = score_sample("v1", "m1", "hello world", "hello big world")
    assert score.insertions == 1
    assert score.wer > 0


def test_score_one_deletion():
    score = score_sample("v1", "m1", "hello world", "hello")
    assert score.deletions == 1


def test_score_empty_strings():
    score = score_sample("v1", "m1", "", "")
    assert score.wer == 0.0
    assert score.cer == 0.0


def test_score_case_insensitive():
    score = score_sample("v1", "m1", "Hello World", "hello world")
    assert score.wer == 0.0


def test_aggregate_scores():
    scores = [
        score_sample("v1__clean", "m1", "hello world", "hello world"),
        score_sample("v1__noise", "m1", "hello world", "hello earth"),
    ]
    agg = aggregate_scores(scores, "m1")
    assert agg.n_samples == 2
    assert agg.macro_wer == 0.25  # (0.0 + 0.5) / 2


def test_aggregate_by_condition():
    scores = [
        score_sample("v1__clean", "m1", "hello world", "hello world"),
        score_sample("v1__noise", "m1", "hello world", "hello earth"),
        score_sample("v2__clean", "m1", "foo bar", "foo bar"),
    ]
    clean = aggregate_scores(scores, "m1", condition_id="clean")
    assert clean.n_samples == 2  # v1__clean and v2__clean
    assert clean.macro_wer == 0.0
