"""Room impulse response convolution for reverb simulation.

Uses torchaudio for RIR loading and convolution.
"""

from __future__ import annotations

import torch
import torchaudio

from ..manifest import TransformParam


def _load_rir(rir_path: str) -> tuple[torch.Tensor, int]:
    """Load RIR file, return (waveform, sample_rate) for torchaudio compat."""
    import soundfile as sf

    info = sf.info(rir_path)
    sr = info.samplerate
    audio, _ = sf.read(rir_path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]  # (1, samples) for mono
    else:
        audio = audio.T  # (channels, samples)
    return torch.from_numpy(audio.copy()).float(), sr


def convolve_rir(
    speech: torch.Tensor,
    rir: torch.Tensor,
    max_rir_samples: int | None = None,
    wet_dry_ratio: float = 0.3,
) -> torch.Tensor:
    """Convolve speech with a room impulse response.

    Mixes dry (original) and wet (reverberant) signals.
    wet_dry_ratio: 0.0 = all dry, 1.0 = all wet
    Output is peak-normalized to prevent clipping.
    """
    # Ensure mono
    if rir.shape[0] > 1:
        rir = rir[0:1]
    if speech.shape[0] > 1:
        speech = speech[0:1]

    # Truncate RIR if too long (prevents garbled output)
    if max_rir_samples and rir.shape[-1] > max_rir_samples:
        rir = rir[..., :max_rir_samples]

    # Normalize RIR to unit energy for consistent reverb level
    rir_rms = torch.sqrt(torch.mean(rir**2))
    if rir_rms > 0:
        rir = rir / rir_rms

    # Convolve in frequency domain
    speech_len = speech.shape[-1]
    rir_len = rir.shape[-1]

    # Pad to avoid circular convolution artifacts
    n_fft = speech_len + rir_len - 1
    # Next power of 2 for efficiency
    n_fft = 2 ** (n_fft - 1).bit_length()

    S = torch.fft.rfft(speech, n=n_fft)
    R = torch.fft.rfft(rir, n=n_fft)
    wet = torch.fft.irfft(S * R, n=n_fft)[..., :speech_len]

    # Mix dry and wet signals
    mixed = (1 - wet_dry_ratio) * speech + wet_dry_ratio * wet

    # Peak normalize
    peak = mixed.abs().max()
    if peak > 0:
        mixed = mixed / peak * 0.95

    return mixed


def apply_reverb_condition(
    speech: torch.Tensor,
    rir_path: str,
    sample_rate: int = 16000,
    max_rir_seconds: float = 5.0,
    wet_dry_ratio: float = 0.3,
) -> tuple[torch.Tensor, TransformParam]:
    """Apply reverb condition using a real room impulse response.

    Args:
        speech: Input audio tensor
        rir_path: Path to RIR file
        sample_rate: Target sample rate
        max_rir_seconds: Maximum RIR duration in seconds (truncates longer RIRs)
        wet_dry_ratio: 0.0 = all dry, 1.0 = all wet (default 0.3)

    Returns:
        (reverberant_waveform, transform_param)
    """
    rir, rir_sr = _load_rir(rir_path)
    if rir_sr != sample_rate:
        rir = torchaudio.functional.resample(rir, rir_sr, sample_rate)

    max_rir_samples = int(max_rir_seconds * sample_rate)
    reverberant = convolve_rir(
        speech, rir, max_rir_samples=max_rir_samples, wet_dry_ratio=wet_dry_ratio
    )

    param = TransformParam(
        type="convolve_rir",
        params={
            "rir_path": rir_path,
            "max_rir_seconds": max_rir_seconds,
            "wet_dry_ratio": wet_dry_ratio,
        },
    )

    return reverberant, param
