"""Download noise and RIR assets from Hugging Face.

MUSAN noise: FluidInference/musan (CC BY 4.0)
RIR dataset: schism-audio/rirs-noises (Apache 2.0)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from datasets import load_dataset


MUSAN_REPO = "FluidInference/musan"
RIR_REPO = "schism-audio/rirs-noises"


def fetch_musan_noise(output_dir: Path, split: str = "train") -> Path:
    """Download MUSAN noise files to output_dir/noise/musan/.

    Returns the path to the noise directory.
    """
    noise_dir = output_dir / "noise" / "musan"
    noise_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(MUSAN_REPO, split=split, streaming=True)

    count = 0
    for sample in ds:
        audio = sample.get("audio")
        if audio is None:
            continue

        # MUSAN samples are stored with path like "noise/free-sound/noise-0001.wav"
        path = sample.get("path", f"noise-{count:04d}.wav")
        dest = noise_dir / Path(path).name

        if not dest.exists():
            # The audio array is already loaded; save it
            import soundfile as sf

            sf.write(str(dest), audio["array"], audio["sampling_rate"])

        count += 1

    print(f"Downloaded {count} MUSAN noise files to {noise_dir}")
    return noise_dir


def fetch_rir_dataset(output_dir: Path) -> Path:
    """Download RIR files to output_dir/rir/rirs-noises/.

    Returns the path to the RIR directory.
    """
    rir_dir = output_dir / "rir" / "rirs-noises"
    rir_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(RIR_REPO, split="train", streaming=True)

    count = 0
    for sample in ds:
        audio = sample.get("audio")
        if audio is None:
            continue

        path = sample.get("path", f"rir-{count:04d}.wav")
        dest = rir_dir / Path(path).name

        if not dest.exists():
            import soundfile as sf

            sf.write(str(dest), audio["array"], audio["sampling_rate"])

        count += 1

    print(f"Downloaded {count} RIR files to {rir_dir}")
    return rir_dir


def fetch_all_assets(output_dir: Path) -> dict[str, Path]:
    """Download all assets (noise + RIRs) to output_dir.

    Returns dict with 'noise' and 'rir' paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_path = fetch_musan_noise(output_dir)
    rir_path = fetch_rir_dataset(output_dir)

    return {"noise": noise_path, "rir": rir_path}
