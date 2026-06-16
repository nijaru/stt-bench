"""Cheap microphone simulation via bandpass filtering and EQ.

Uses scipy for filtering; falls back to basic torchaudio operations.
"""

from __future__ import annotations

import torch
import torchaudio

from ..manifest import TransformParam


def apply_bandpass(
    waveform: torch.Tensor,
    low_freq: float = 200.0,
    high_freq: float = 6000.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Apply bandpass filter to simulate frequency-limited mic."""
    try:
        import numpy as np
        from scipy.signal import butter, sosfilt

        nyq = sample_rate / 2
        low = low_freq / nyq
        high = min(high_freq / nyq, 0.99)

        sos = butter(4, [low, high], btype="band", output="sos")
        audio_np = waveform.squeeze().numpy().astype(np.float32)
        filtered = sosfilt(sos, audio_np)
        return torch.from_numpy(filtered).unsqueeze(0).to(waveform.dtype)

    except ImportError:
        # Fallback: simple highpass via differentiation + lowpass via moving average
        # Less accurate but no scipy dependency
        highpass = torchaudio.functional.highpass_biquad(waveform, sample_rate, low_freq)
        lowpass = torchaudio.functional.lowpass_biquad(highpass, sample_rate, high_freq)
        return lowpass


def apply_cheap_mic(
    speech: torch.Tensor,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, TransformParam]:
    """Simulate a cheap electret microphone.

    Applies bandpass (200-6000Hz) and slight mid-frequency dip.
    """
    # Bandpass to simulate cheap mic frequency response
    filtered = apply_bandpass(speech, low_freq=200.0, high_freq=6000.0, sample_rate=sample_rate)

    # Apply slight mid-frequency dip (characteristic of cheap mics)
    # Use a peaking EQ at ~2kHz with negative gain
    try:
        import numpy as np
        from scipy.signal import butter, sosfilt

        nyq = sample_rate / 2
        # Simple notch at 2kHz
        sos_notch = butter(2, [1800 / nyq, 2200 / nyq], btype="bandstop", output="sos")
        audio_np = filtered.squeeze().numpy().astype(np.float32)
        mid_dipped = sosfilt(sos_notch, audio_np)
        # Blend: 70% filtered + 30% with dip
        result_np = 0.7 * audio_np + 0.3 * mid_dipped
        result = torch.from_numpy(result_np).unsqueeze(0).to(speech.dtype)
    except ImportError:
        result = filtered

    param = TransformParam(
        type="bandpass_eq",
        params={
            "low_freq_hz": 200,
            "high_freq_hz": 6000,
            "mid_dip_center_hz": 2000,
            "mid_dip_depth_db": -6,
        },
    )

    return result, param
