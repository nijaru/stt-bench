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


def apply_mic_profile(
    speech: torch.Tensor,
    mic_type: str = "phone",
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, TransformParam]:
    """Apply microphone frequency response profile.

    Mic types:
    - phone: smartphone mic (decent quality, limited low-end)
    - laptop: laptop internal mic (cheap, resonant, mono)
    """
    if mic_type == "phone":
        # Smartphone mic: decent quality, slight low-end rolloff
        # -3dB at 100Hz, flat mids, slight high-freq rolloff above 12kHz
        low_freq = 100.0
        high_freq = 12000.0
        mid_dip_center = None  # No mid dip for phone
    else:  # laptop
        # Laptop internal mic: cheap, resonant, mono
        # -3dB at 200Hz, -6dB at 100Hz, resonant peak at 3-4kHz, rolloff above 8kHz
        low_freq = 200.0
        high_freq = 8000.0
        mid_dip_center = 3500.0  # Resonant peak

    # Apply bandpass
    filtered = apply_bandpass(
        speech, low_freq=low_freq, high_freq=high_freq, sample_rate=sample_rate,
    )

    # Apply mid-frequency resonance if specified
    if mid_dip_center:
        try:
            import numpy as np
            from scipy.signal import butter, sosfilt

            nyq = sample_rate / 2
            # Peaking EQ for resonance
            freq_low = mid_dip_center * 0.8
            freq_high = mid_dip_center * 1.2
            sos_peak = butter(2, [freq_low / nyq, freq_high / nyq], btype="band", output="sos")
            audio_np = filtered.squeeze().numpy().astype(np.float32)
            peaked = sosfilt(sos_peak, audio_np)
            # Blend: 80% filtered + 20% with peak
            result_np = 0.8 * audio_np + 0.2 * peaked
            result = torch.from_numpy(result_np).unsqueeze(0).to(speech.dtype)
        except ImportError:
            result = filtered
    else:
        result = filtered

    param = TransformParam(
        type="mic_profile",
        params={
            "mic_type": mic_type,
            "low_freq_hz": low_freq,
            "high_freq_hz": high_freq,
            "mid_resonance_hz": mid_dip_center,
        },
    )

    return result, param
