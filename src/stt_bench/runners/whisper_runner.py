"""Whisper model runner using transformers pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..manifest import ConditionVariant, Hypothesis
from . import BaseRunner, register_runner


@register_runner("openai/whisper")
class WhisperRunner(BaseRunner):
    """Runner for Whisper models via Hugging Face transformers."""

    name = "whisper_transformers"
    version = "0.1.0"

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "auto", **kwargs):
        super().__init__(model_id=model_id, device=device, **kwargs)
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        device = self._resolve_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
            torch_dtype=dtype,
        )

    def transcribe(self, variant: ConditionVariant) -> Hypothesis:
        """Transcribe a condition variant using Whisper."""
        self._load_pipeline()

        audio_path = Path(variant.source_uri)
        if not audio_path.exists():
            # Try standard generated location
            audio_path = Path(f"data/generated/{variant.variant_id}.wav")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {variant.source_uri}")

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        start = time.monotonic()
        result = self._pipeline(audio)
        elapsed = time.monotonic() - start

        text = result.get("text", "") if isinstance(result, dict) else str(result)

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=text.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=round(elapsed, 3),
        )
