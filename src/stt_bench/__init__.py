"""STT-Bench: Real-world robustness benchmarks for speech-to-text systems."""

__version__ = "0.1.0"

from .manifest import (
    ConditionVariant,
    Hypothesis,
    SourceClip,
    TransformParam,
    iter_manifest,
    read_manifest,
    write_manifest,
)
from .scoring.score import SampleScore

__all__ = [
    "ConditionVariant",
    "Hypothesis",
    "SampleScore",
    "SourceClip",
    "TransformParam",
    "iter_manifest",
    "read_manifest",
    "write_manifest",
]
