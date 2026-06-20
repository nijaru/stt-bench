"""Audio I/O helpers using soundfile.

Avoids torchaudio.load/save which require torchcodec (unavailable on macOS ARM64).
All functions return torch tensors in (channels, samples) format at the target sample rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str | Path, target_sr: int | None = None) -> torch.Tensor:
    """Load audio file as torch tensor in (channels, samples) format.

    Args:
        path: Path to audio file (wav, flac, mp3, etc.)
        target_sr: If specified, resample to this rate using librosa.

    Returns:
        torch.Tensor of shape (channels, samples), float32.
    """
    path = str(path)

    if target_sr is not None and target_sr != _get_sr(path):
        # Use librosa for resampling on load
        import librosa

        audio, sr = librosa.load(path, sr=target_sr, mono=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        return torch.from_numpy(audio.copy()).float()

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T  # soundfile returns (samples, channels)
    return torch.from_numpy(audio.copy()).float()


def _get_sr(path: str) -> int:
    """Get sample rate of audio file."""
    info = sf.info(path)
    return info.samplerate


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save torch tensor to audio file.

    Args:
        path: Output path (.wav or .flac)
        waveform: torch.Tensor of shape (channels, samples)
        sample_rate: Sample rate
    """
    path = str(path)
    audio = waveform.squeeze().numpy()
    if audio.ndim == 0:
        audio = audio[np.newaxis]
    sf.write(path, audio, sample_rate)
