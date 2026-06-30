"""Select source clips from LibriSpeech test-clean.

Uses Hugging Face datasets-server API to get metadata, then measures
actual audio duration from downloaded files for accurate filtering.
Writes SourceClip manifest as JSONL.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any

import requests

from ..manifest import SourceClip, write_manifest

LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_CONFIG = "clean"
LIBRISPEECH_SPLIT = "test"
ROWS_PER_PAGE = 100

# Word-count duration estimate is approximate (LibriSpeech read speech
# varies 130-170 wpm).  We use a generous margin when filtering so we
# don't exclude valid clips, then measure actual duration in a second pass.
_WPM_ESTIMATE = 150  # words per minute
_WPS_ESTIMATE = _WPM_ESTIMATE / 60  # words per second (2.5)


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
    rows = []
    for item in data.get("rows", []):
        row = item["row"]
        row["_row_idx"] = item.get("row_idx", offset + len(rows))
        rows.append(row)
    total = data.get("num_rows_total", 0)
    return rows, total


def _row_to_candidate(
    row: dict[str, Any],
    min_duration: float,
    max_duration: float,
) -> dict | None:
    """Convert an API row to a candidate clip dict, or None if filtered out.

    Uses word-count estimate with generous margins as a first pass.
    Actual duration is measured later.
    """
    text = row.get("text", "").strip()
    if not text:
        return None

    # Generous word-count filter (actual measurement follows)
    word_count = len(text.split())
    est_duration = word_count / _WPS_ESTIMATE
    margin = 0.4  # tolerate ±40% estimation error
    if not (min_duration * (1 - margin) <= est_duration <= max_duration * (1 + margin)):
        return None

    row_idx = row.get("_row_idx")
    if row_idx is None:
        return None

    # Capture the signed audio URL for duration measurement
    audio_list = row.get("audio", [])
    audio_src_url = audio_list[0]["src"] if audio_list else None

    audio_uri = f"hf://{LIBRISPEECH_REPO}/{LIBRISPEECH_CONFIG}/{LIBRISPEECH_SPLIT}/{row_idx}"

    return {
        "audio_uri": audio_uri,
        "audio_src_url": audio_src_url,
        "clip_id": row.get("id", ""),
        "reference_text": text,
        "speaker_id": str(row.get("speaker_id", "unknown")),
        "estimated_duration": est_duration,
        "sample_rate": 16000,
    }


def _measure_audio_duration(src_url: str) -> float:
    """Download an audio file and return its actual duration in seconds."""
    import soundfile as sf

    response = requests.get(src_url, timeout=60)
    response.raise_for_status()
    info = sf.info(io.BytesIO(response.content))
    return info.duration


def _verify_candidate_durations(
    candidates: list[dict],
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    """Measure actual durations for candidates and filter by duration bounds."""
    verified = []
    for c in candidates:
        src_url = c.get("audio_src_url")
        if src_url is None:
            continue
        try:
            actual = _measure_audio_duration(src_url)
        except Exception:
            continue
        if min_duration <= actual <= max_duration:
            c["duration"] = actual
            verified.append(c)
    return verified


def select_librispeech_clips(
    n_clips: int = 30,
    min_duration: float = 10.0,
    max_duration: float = 30.0,
    seed: int = 42,
) -> list[SourceClip]:
    """Select diverse clips from LibriSpeech test-clean.

    Paginates through the dataset via API, filters by estimated then
    actual duration, and selects clips with speaker diversity.
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

        # Early exit: enough candidates with good speaker diversity
        if len(candidates) >= n_clips * 8 and len(seen_speakers) >= n_clips * 3:
            break

    print(
        f"Collected {len(candidates)} candidates from {len(seen_speakers)} speakers "
        f"(word-count filter)"
    )

    # Second pass: measure actual durations
    print("Measuring actual durations...")
    verified = _verify_candidate_durations(candidates, min_duration, max_duration)
    print(f"  {len(verified)} passed actual duration check ({min_duration}-{max_duration}s)")

    if len(verified) < n_clips:
        print(
            f"Warning: only {len(verified)} candidates with actual durations in range "
            f"(requested {n_clips})"
        )

    # Select clips with speaker diversity
    rng.shuffle(verified)

    selected: list[dict] = []
    speakers_used: set[str] = set()
    remaining: list[dict] = []

    # First pass: one clip per speaker
    for c in verified:
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
            audio_uri=c["audio_uri"],
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
