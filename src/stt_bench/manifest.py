"""Manifest schemas for STT-Bench pipeline.

Manifests are JSONL files. Each line is one record. Schemas validate on read/write.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class SourceClip:
    """A source speech clip with reference transcript."""

    clip_id: str
    audio_uri: str  # Hugging Face dataset path (hf://dataset/split/index)
    reference_text: str
    license: str
    source_dataset: str
    duration_seconds: float
    sample_rate: int = 16000
    channels: int = 1
    speaker_id: str | None = None
    accent: str | None = None
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> SourceClip:
        return cls(**json.loads(line))


@dataclass
class TransformParam:
    """Parameters for a single audio transform step."""

    type: str  # "add_background_noise", "convolve_rir", "codec", "bandpass_eq"
    params: dict = field(default_factory=dict)
    snr_achieved_db: float | None = None  # measured SNR for noise mixing
    seed: int | None = None


@dataclass
class ConditionVariant:
    """A condition variant of a source clip."""

    variant_id: str
    clip_id: str
    condition_id: str
    source_uri: str
    reference_text: str
    transforms: list[TransformParam] = field(default_factory=list)
    checksum_sha256: str | None = None
    sample_rate: int = 16000
    duration_seconds: float | None = None
    noise_asset_id: str | None = None
    rir_asset_id: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["transforms"] = [asdict(t) for t in self.transforms]
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> ConditionVariant:
        data = json.loads(line)
        data["transforms"] = [TransformParam(**t) for t in data.get("transforms", [])]
        return cls(**data)


@dataclass
class Hypothesis:
    """A model's transcription of a condition variant."""

    variant_id: str
    model_id: str
    hypothesis_text: str
    runner: str
    runner_version: str
    runtime_seconds: float | None = None
    model_revision: str | None = None
    started_at: str | None = None
    config_hash: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Hypothesis:
        return cls(**json.loads(line))


@dataclass
class SampleScore:
    """Score for a single hypothesis."""

    variant_id: str
    model_id: str
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    ref_normalized: str
    hyp_normalized: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> SampleScore:
        return cls(**json.loads(line))


# --- Manifest I/O ---


def read_manifest(path: Path, cls: type) -> list:
    """Read a JSONL manifest file into a list of dataclass instances."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(cls.from_json(line))
    return items


def iter_manifest(path: Path, cls: type) -> Iterator:
    """Iterate over a JSONL manifest file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield cls.from_json(line)


def write_manifest(path: Path, items: list) -> None:
    """Write a list of dataclass instances to a JSONL manifest file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(item.to_json() + "\n")
