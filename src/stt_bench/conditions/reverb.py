"""Room impulse response convolution for reverb simulation.

Uses torchaudio for RIR loading and convolution.
"""

from __future__ import annotations

import torch
import torchaudio

from ..manifest import TransformParam


def convolve_rir(
    speech: torch.Tensor,
    rir: torch.Tensor,
) -> torch.Tensor:
    """Convolve speech with a room impulse response.

    Output is peak-normalized to prevent clipping.
    """
    # Ensure mono
    if rir.shape[0] > 1:
        rir = rir[0:1]
    if speech.shape[0] > 1:
        speech = speech[0:1]

    # Convolve in frequency domain
    speech_len = speech.shape[-1]
    rir_len = rir.shape[-1]

    # Pad to avoid circular convolution artifacts
    n_fft = speech_len + rir_len - 1
    # Next power of 2 for efficiency
    n_fft = 2 ** (n_fft - 1).bit_length()

    S = torch.fft.rfft(speech, n=n_fft)
    R = torch.fft.rfft(rir, n=n_fft)
    convolved = torch.fft.irfft(S * R, n=n_fft)[..., :speech_len]

    # Peak normalize
    peak = convolved.abs().max()
    if peak > 0:
        convolved = convolved / peak * 0.95

    return convolved


def apply_reverb_condition(
    speech: torch.Tensor,
    rir_path: str,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, TransformParam]:
    """Apply reverb condition using a real room impulse response.

    Returns (reverberant_waveform, transform_param).
    """
    rir, rir_sr = torchaudio.load(rir_path)
    if rir_sr != sample_rate:
        rir = torchaudio.functional.resample(rir, rir_sr, sample_rate)

    reverberant = convolve_rir(speech, rir)

    param = TransformParam(
        type="convolve_rir",
        params={"rir_path": rir_path},
    )

    return reverberant, param
