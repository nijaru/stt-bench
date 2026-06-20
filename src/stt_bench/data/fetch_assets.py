"""Download noise and RIR assets from Hugging Face.

Uses the datasets-server API to get file URLs without audio decoding.
MUSAN noise: FluidInference/musan (CC BY 4.0) — cafe, traffic, general noise
RIR dataset: schism-audio/rirs-noises (Apache 2.0) — real room impulse responses
"""

from __future__ import annotations

from pathlib import Path

import requests

MUSAN_REPO = "FluidInference/musan"
RIR_REPO = "schism-audio/rirs-noises"
ROWS_PER_PAGE = 100


def _fetch_rows(repo: str, offset: int = 0, length: int = ROWS_PER_PAGE) -> tuple[list[dict], int]:
    """Fetch a page of rows from the HF datasets-server API."""
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": repo,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    rows = [item["row"] for item in data.get("rows", [])]
    total = data.get("num_rows_total", 0)
    return rows, total


def _download_audio(url: str, dest: Path, label: str | None = None) -> None:
    """Download an audio file from a URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)


def fetch_musan_noise(
    output_dir: Path,
    n_files: int = 50,
) -> Path:
    """Download MUSAN noise files to output_dir/noise/musan/.

    Downloads up to n_files noise recordings.
    Returns the path to the noise directory.
    """
    noise_dir = output_dir / "noise" / "musan"
    noise_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    offset = 0
    total = None

    while total is None or offset < total:
        rows, total = _fetch_rows(MUSAN_REPO, offset=offset)
        if not rows:
            break

        for row in rows:
            if count >= n_files:
                break

            audio_list = row.get("audio", [])
            if not audio_list:
                continue

            audio_url = audio_list[0]["src"]
            label = row.get("label", "unknown")
            dest = noise_dir / f"musan_{count:04d}_{label}.flac"

            if not dest.exists():
                _download_audio(audio_url, dest)
            count += 1

        if count >= n_files:
            break
        offset += len(rows)

    print(f"Downloaded {count} MUSAN noise files to {noise_dir}")
    return noise_dir


def fetch_rir_dataset(
    output_dir: Path,
    n_files: int = 20,
) -> Path:
    """Download RIR files to output_dir/rir/rirs-noises/.

    Downloads up to n_files room impulse responses.
    Returns the path to the RIR directory.
    """
    rir_dir = output_dir / "rir" / "rirs-noises"
    rir_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    offset = 0
    total = None

    while total is None or offset < total:
        rows, total = _fetch_rows(RIR_REPO, offset=offset)
        if not rows:
            break

        for row in rows:
            if count >= n_files:
                break

            audio_list = row.get("audio", [])
            if not audio_list:
                continue

            audio_url = audio_list[0]["src"]
            dest = rir_dir / f"rir_{count:04d}.flac"

            if not dest.exists():
                _download_audio(audio_url, dest)
            count += 1

        if count >= n_files:
            break
        offset += len(rows)

    print(f"Downloaded {count} RIR files to {rir_dir}")
    return rir_dir


def fetch_all_assets(output_dir: Path, n_noise: int = 50, n_rir: int = 20) -> dict[str, Path]:
    """Download all assets (noise + RIRs) to output_dir.

    Returns dict with 'noise' and 'rir' paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_path = fetch_musan_noise(output_dir, n_files=n_noise)
    rir_path = fetch_rir_dataset(output_dir, n_files=n_rir)

    return {"noise": noise_path, "rir": rir_path}
