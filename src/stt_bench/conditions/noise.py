"""Additive noise mixing with real noise recordings.

Uses torchaudio for noise loading and SNR-calibrated mixing.
"""

from __future__ import annotations

import random

import torch
import torchaudio

from ..manifest import TransformParam


def _load_noise(noise_path: str) -> tuple[torch.Tensor, int]:
    """Load noise file, return (waveform, sample_rate) for torchaudio compat."""
    import soundfile as sf

    info = sf.info(noise_path)
    sr = info.samplerate
    audio, _ = sf.read(noise_path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]  # (1, samples) for mono
    else:
        audio = audio.T  # (channels, samples)
    return torch.from_numpy(audio.copy()).float(), sr


def compute_rms(waveform: torch.Tensor) -> float:
    """Compute RMS amplitude of a waveform."""
    return float(torch.sqrt(torch.mean(waveform**2)))


def mix_at_snr(
    speech: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Mix speech and noise at target SNR.

    Returns (mixed_waveform, achieved_snr_db).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Trim noise to speech length
    if noise.shape[-1] > speech.shape[-1]:
        start = random.randint(0, noise.shape[-1] - speech.shape[-1])
        noise = noise[..., start : start + speech.shape[-1]]
    elif noise.shape[-1] < speech.shape[-1]:
        # Loop noise
        repeats = (speech.shape[-1] // noise.shape[-1]) + 1
        noise = noise.repeat(1, repeats)[..., : speech.shape[-1]]

    speech_rms = compute_rms(speech)
    noise_rms = compute_rms(noise)

    if noise_rms == 0:
        return speech.clone(), float("inf")

    # Scale noise to achieve target SNR
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = speech + noise_scaled

    # Measure achieved SNR
    residual_noise = mixed - speech
    achieved_snr = 20 * torch.log10(
        torch.tensor(speech_rms / (compute_rms(residual_noise) + 1e-10))
    )

    return mixed, float(achieved_snr)


def apply_noise_condition(
    speech: torch.Tensor,
    noise_path: str,
    snr_db: float,
    sample_rate: int = 16000,
    seed: int | None = None,
) -> tuple[torch.Tensor, TransformParam]:
    """Apply additive noise condition.

    Returns (mixed_waveform, transform_param).
    """
    noise, noise_sr = _load_noise(noise_path)
    if noise_sr != sample_rate:
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)

    # Mix on first channel if multi-channel
    if noise.shape[0] > 1:
        noise = noise[0:1]

    mixed, achieved_snr = mix_at_snr(speech, noise, snr_db, seed=seed)

    param = TransformParam(
        type="add_background_noise",
        params={"noise_path": noise_path, "snr_target_db": snr_db},
        snr_achieved_db=round(achieved_snr, 2),
        seed=seed,
    )

    return mixed, param
