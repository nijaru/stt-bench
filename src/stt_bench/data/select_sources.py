"""Select source clips from LibriSpeech test-clean.

Uses Hugging Face datasets-server API to get metadata without audio decoding.
Writes SourceClip manifest as JSONL.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import requests

from ..manifest import SourceClip, write_manifest

LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_CONFIG = "clean"
LIBRISPEECH_SPLIT = "test"
ROWS_PER_PAGE = 100


def _fetch_rows(offset: int = 0, length: int = ROWS_PER_PAGE) -> tuple[list[dict[str, Any]], int]:
    """Fetch a page of rows from the HF datasets-server API.

    Returns (rows, total_count). Rows include text, speaker_id, id, audio URL.
    """
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": LIBRISPEECH_REPO,
        "config": LIBRISPEECH_CONFIG,
        "split": LIBRISPEECH_SPLIT,
        "offset": offset,
        "length": length,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    rows = [item["row"] for item in data.get("rows", [])]
    total = data.get("num_rows_total", 0)
    return rows, total


def _row_to_candidate(row: dict[str, Any], min_duration: float, max_duration: float) -> dict | None:
    """Convert an API row to a candidate clip dict, or None if filtered out."""
    text = row.get("text", "").strip()
    if not text:
        return None

    # LibriSpeech is read speech at ~140-160 wpm. Estimate duration from word count.
    word_count = len(text.split())
    duration = word_count / 2.5  # ~150 wpm = 2.5 wps

    if not (min_duration <= duration <= max_duration):
        return None

    # Extract audio URL from the API response
    audio_list = row.get("audio", [])
    audio_url = audio_list[0]["src"] if audio_list else None

    return {
        "audio_url": audio_url,
        "clip_id": row.get("id", ""),
        "reference_text": text,
        "speaker_id": str(row.get("speaker_id", "unknown")),
        "duration": duration,
        "sample_rate": 16000,
    }


def select_librispeech_clips(
    n_clips: int = 30,
    min_duration: float = 10.0,
    max_duration: float = 30.0,
    seed: int = 42,
) -> list[SourceClip]:
    """Select diverse clips from LibriSpeech test-clean.

    Paginates through the dataset via API, filters by estimated duration,
    and selects clips with speaker diversity.
    """
    rng = random.Random(seed)

    candidates: list[dict] = []
    seen_speakers: set[str] = set()

    offset = 0
    total = None
    while total is None or offset < total:
        rows, total = _fetch_rows(offset=offset, length=ROWS_PER_PAGE)
        if not rows:
            break

        for row in rows:
            candidate = _row_to_candidate(row, min_duration, max_duration)
            if candidate:
                candidates.append(candidate)
                seen_speakers.add(candidate["speaker_id"])

        offset += len(rows)

        # Early exit if we have enough candidates with good speaker diversity
        if len(candidates) >= n_clips * 5 and len(seen_speakers) >= n_clips * 2:
            break

    if len(candidates) < n_clips:
        print(f"Warning: only {len(candidates)} candidates found (requested {n_clips})")

    # Select clips with speaker diversity
    rng.shuffle(candidates)

    selected: list[dict] = []
    speakers_used: set[str] = set()
    remaining: list[dict] = []

    # First pass: one clip per speaker
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
            audio_uri=c["audio_url"] or "",
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
