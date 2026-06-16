"""STT-Bench: Real-world robustness benchmarks for speech-to-text systems."""

__version__ = "0.1.0"

from .manifest import (
    ConditionVariant,
    Hypothesis,
    SampleScore,
    SourceClip,
    TransformParam,
    iter_manifest,
    read_manifest,
    write_manifest,
)

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
