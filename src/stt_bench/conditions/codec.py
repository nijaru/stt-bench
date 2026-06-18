"""Codec simulation: telephony (G.711 mu-law) and low-bitrate MP3.

Uses torchaudio for mu-law and pedalboard for MP3 when available,
falls back to pure-Python implementations.
"""

from __future__ import annotations

import torch
import torchaudio

from ..manifest import TransformParam


def apply_mulaw_codec(
    speech: torch.Tensor,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, TransformParam]:
    """Simulate G.711 mu-law telephony codec.

    Downsamples to 8kHz, applies mu-law companding, upsamples back.
    """
    # Downsample to 8kHz (telephony)
    speech_8k = torchaudio.functional.resample(speech, sample_rate, 8000)

    # Apply mu-law companding (256 levels, matching G.711)
    mu = 255
    speech_abs = speech_8k.abs()
    encoded = torch.sign(speech_8k) * torch.log1p(mu * speech_abs) / torch.log1p(
        torch.tensor(float(mu))
    )

    # Quantize to 8-bit
    encoded = torch.round(encoded * 127.5 + 127.5) / 127.5 - 1.0

    # Upsample back to original sample rate
    decoded = torchaudio.functional.resample(encoded, 8000, sample_rate)

    param = TransformParam(
        type="codec",
        params={"codec": "g711_mulaw", "telephony_sample_rate": 8000},
    )

    return decoded, param


def apply_mp3_codec(
    speech: torch.Tensor,
    sample_rate: int = 16000,
    bitrate_kbps: int = 32,
) -> tuple[torch.Tensor, TransformParam]:
    """Simulate low-bitrate MP3 compression.

    Uses pedalboard if available, falls back to basic resampling degradation.
    """
    try:
        import numpy as np
        from pedalboard import MP3Compressor, Pedalboard

        # Convert to numpy for pedalboard
        audio_np = speech.squeeze().numpy().astype(np.float32)
        board = Pedalboard([MP3Compressor(bitrate_kbps=bitrate_kbps)])
        processed = board(audio_np, sample_rate)

        result = torch.from_numpy(processed).unsqueeze(0)
        if result.shape[-1] != speech.shape[-1]:
            result = result[..., : speech.shape[-1]]

        param = TransformParam(
            type="codec",
            params={"codec": "mp3", "bitrate_kbps": bitrate_kbps},
        )
        return result, param

    except ImportError:
        # Fallback: simulate low-bitrate with aggressive resampling
        # Not a perfect MP3 simulation, but captures bandwidth limitation
        low_sr = max(8000, sample_rate * bitrate_kbps // 128)
        low = torchaudio.functional.resample(speech, sample_rate, low_sr)
        result = torchaudio.functional.resample(low, low_sr, sample_rate)

        if result.shape[-1] != speech.shape[-1]:
            result = result[..., : speech.shape[-1]]

        param = TransformParam(
            type="codec",
            params={
                "codec": "mp3_approximate",
                "bitrate_kbps": bitrate_kbps,
                "fallback": True,
            },
        )
        return result, param


def apply_opus_codec(
    speech: torch.Tensor,
    sample_rate: int = 16000,
    bitrate_kbps: int = 6,
) -> tuple[torch.Tensor, TransformParam]:
    """Simulate low-bitrate Opus compression.

    Opus is commonly used for voice messages and low-bandwidth VoIP.
    Uses pedalboard if available, falls back to bandwidth limitation.
    """
    try:
        import numpy as np
        from pedalboard import Pedalboard
        from pedalboard.opus import OpusCompressor

        audio_np = speech.squeeze().numpy().astype(np.float32)
        board = Pedalboard([OpusCompressor(bitrate_kbps=bitrate_kbps)])
        processed = board(audio_np, sample_rate)

        result = torch.from_numpy(processed).unsqueeze(0)
        if result.shape[-1] != speech.shape[-1]:
            result = result[..., : speech.shape[-1]]

        param = TransformParam(
            type="codec",
            params={"codec": "opus", "bitrate_kbps": bitrate_kbps},
        )
        return result, param

    except (ImportError, AttributeError):
        # Fallback: simulate low-bitrate Opus with bandwidth limitation
        # Opus at 6kbps has very limited bandwidth (~4kHz)
        low_sr = 8000
        low = torchaudio.functional.resample(speech, sample_rate, low_sr)
        result = torchaudio.functional.resample(low, low_sr, sample_rate)

        if result.shape[-1] != speech.shape[-1]:
            result = result[..., : speech.shape[-1]]

        param = TransformParam(
            type="codec",
            params={
                "codec": "opus_approximate",
                "bitrate_kbps": bitrate_kbps,
                "fallback": True,
            },
        )
        return result, param


def apply_aac_codec(
    speech: torch.Tensor,
    sample_rate: int = 16000,
    bitrate_kbps: int = 32,
) -> tuple[torch.Tensor, TransformParam]:
    """Simulate low-bitrate AAC compression.

    AAC is commonly used for video conferencing (Zoom, Teams, etc.).
    Uses pedalboard if available, falls back to bandwidth limitation.
    """
    try:
        import numpy as np
        from pedalboard import AACCompressor, Pedalboard

        audio_np = speech.squeeze().numpy().astype(np.float32)
        board = Pedalboard([AACCompressor(bitrate_kbps=bitrate_kbps)])
        processed = board(audio_np, sample_rate)

        result = torch.from_numpy(processed).unsqueeze(0)
        if result.shape[-1] != speech.shape[-1]:
            result = result[..., : speech.shape[-1]]

        param = TransformParam(
            type="codec",
            params={"codec": "aac", "bitrate_kbps": bitrate_kbps},
        )
        return result, param

    except (ImportError, AttributeError):
        # Fallback: simulate low-bitrate AAC with bandwidth limitation
        # AAC at 32kbps has reasonable quality but some artifacts
        low_sr = 16000  # AAC maintains sample rate but reduces quality
        low = torchaudio.functional.resample(speech, sample_rate, low_sr)
        result = torchaudio.functional.resample(low, low_sr, sample_rate)

        if result.shape[-1] != speech.shape[-1]:
            result = result[..., : speech.shape[-1]]

        param = TransformParam(
            type="codec",
            params={
                "codec": "aac_approximate",
                "bitrate_kbps": bitrate_kbps,
                "fallback": True,
            },
        )
        return result, param
