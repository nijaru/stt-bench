"""WER/CER scoring using jiwer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import jiwer

from .normalize import normalize


@dataclass
class SampleScore:
    """Score for a single hypothesis against its reference."""

    variant_id: str
    model_id: str
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    ref_normalized: str
    hyp_normalized: str
    ref_raw: str = ""
    hyp_raw: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> SampleScore:
        return cls(**json.loads(line))


@dataclass
class AggregateScore:
    """Aggregated scores across samples."""

    model_id: str
    condition_id: str | None  # None = all conditions
    n_samples: int
    macro_wer: float
    macro_cer: float
    mean_wer: float
    mean_cer: float
    worst_wer: float
    worst_cer: float
    total_insertions: int
    total_deletions: int
    total_substitutions: int


def score_sample(
    variant_id: str,
    model_id: str,
    reference: str,
    hypothesis: str,
) -> SampleScore:
    """Score a single hypothesis against its reference."""
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    # jiwer requires non-empty strings
    if not ref_norm and not hyp_norm:
        return SampleScore(
            variant_id=variant_id,
            model_id=model_id,
            wer=0.0,
            cer=0.0,
            insertions=0,
            deletions=0,
            substitutions=0,
            ref_normalized=ref_norm,
            hyp_normalized=hyp_norm,
            ref_raw=reference,
            hyp_raw=hypothesis,
        )

    result = jiwer.process_words(ref_norm, hyp_norm)
    cer_result = jiwer.process_characters(ref_norm, hyp_norm)

    return SampleScore(
        variant_id=variant_id,
        model_id=model_id,
        wer=result.wer,
        cer=cer_result.cer,
        insertions=result.insertions,
        deletions=result.deletions,
        substitutions=result.substitutions,
        ref_normalized=ref_norm,
        hyp_normalized=hyp_norm,
        ref_raw=reference,
        hyp_raw=hypothesis,
    )


def aggregate_scores(
    scores: list[SampleScore],
    model_id: str,
    condition_id: str | None = None,
) -> AggregateScore:
    """Aggregate sample scores into summary statistics."""
    if condition_id is not None:
        scores = [s for s in scores if s.variant_id.endswith(f"__{condition_id}")]

    if not scores:
        return AggregateScore(
            model_id=model_id,
            condition_id=condition_id,
            n_samples=0,
            macro_wer=0.0,
            macro_cer=0.0,
            mean_wer=0.0,
            mean_cer=0.0,
            worst_wer=0.0,
            worst_cer=0.0,
            total_insertions=0,
            total_deletions=0,
            total_substitutions=0,
        )

    wers = [s.wer for s in scores]
    cers = [s.cer for s in scores]

    return AggregateScore(
        model_id=model_id,
        condition_id=condition_id,
        n_samples=len(scores),
        macro_wer=sum(wers) / len(wers),
        macro_cer=sum(cers) / len(cers),
        mean_wer=sum(wers) / len(wers),
        mean_cer=sum(cers) / len(cers),
        worst_wer=max(wers),
        worst_cer=max(cers),
        total_insertions=sum(s.insertions for s in scores),
        total_deletions=sum(s.deletions for s in scores),
        total_substitutions=sum(s.substitutions for s in scores),
    )
