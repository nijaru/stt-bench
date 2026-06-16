"""Select source clips from LibriSpeech test-clean.

Streams from Hugging Face, filters by duration (10-30s), selects diverse speakers.
Writes SourceClip manifest as JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from ..manifest import SourceClip, write_manifest


LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_SPLIT = "test"


def select_librispeech_clips(
    n_clips: int = 30,
    min_duration: float = 10.0,
    max_duration: float = 30.0,
    seed: int = 42,
) -> list[SourceClip]:
    """Stream LibriSpeech test-clean and select diverse clips.

    Returns n_clips SourceClip instances with duration in [min_duration, max_duration].
    Prioritizes unique speakers for diversity.
    """
    import random

    rng = random.Random(seed)

    ds = load_dataset(
        LIBRISPEECH_REPO,
        "clean",
        split=LIBRISPEECH_SPLIT,
        streaming=True,
    )

    # Collect candidate clips
    candidates: list[dict] = []
    seen_speakers: set[str] = set()

    for sample in ds:
        audio = sample.get("audio", {})
        duration = audio.get("array", [])
        if duration:
            duration = len(duration) / audio.get("sampling_rate", 16000)
        else:
            continue

        if not (min_duration <= duration <= max_duration):
            continue

        speaker_id = str(sample.get("speaker_id", "unknown"))
        text = sample.get("text", "").strip()
        if not text:
            continue

        candidates.append({
            "audio_path": sample.get("path", ""),
            "reference_text": text,
            "speaker_id": speaker_id,
            "duration": duration,
            "sample_rate": audio.get("sampling_rate", 16000),
        })

        seen_speakers.add(speaker_id)

    if len(candidates) < n_clips:
        print(f"Warning: only {len(candidates)} candidates found (requested {n_clips})")

    # Select clips with speaker diversity
    # First pass: one clip per speaker
    selected: list[dict] = []
    remaining: list[dict] = []
    speakers_used: set[str] = set()

    rng.shuffle(candidates)
    for c in candidates:
        if c["speaker_id"] not in speakers_used and len(selected) < n_clips:
            selected.append(c)
            speakers_used.add(c["speaker_id"])
        else:
            remaining.append(c)

    # Second pass: fill remaining slots
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= n_clips:
            break
        selected.append(c)

    # Convert to SourceClip
    clips = []
    for i, c in enumerate(selected):
        clip = SourceClip(
            clip_id=f"librispeech-{i:03d}",
            audio_uri=f"hf://{LIBRISPEECH_REPO}/clean/{LIBRISPEECH_SPLIT}",
            reference_text=c["reference_text"],
            license="CC-BY-4.0",
            source_dataset="librispeech-test-clean",
            duration_seconds=c["duration"],
            sample_rate=c["sample_rate"],
            channels=1,
            speaker_id=c["speaker_id"],
        )
        clips.append(clip)

    return clips


def write_source_manifest(clips: list[SourceClip], output_path: Path) -> None:
    """Write source clips to a JSONL manifest."""
    write_manifest(output_path, clips)
    print(f"Wrote {len(clips)} source clips to {output_path}")
