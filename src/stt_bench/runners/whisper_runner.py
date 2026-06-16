"""Whisper model runner using transformers pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torchaudio

from ..manifest import ConditionVariant, Hypothesis
from . import RunnerProtocol, register_runner


@register_runner("openai/whisper")
class WhisperRunner(RunnerProtocol):
    """Runner for Whisper models via Hugging Face transformers."""

    name = "whisper_transformers"

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "auto"):
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self._pipeline = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        from transformers import pipeline

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

    def transcribe(self, variant: ConditionVariant) -> Hypothesis:
        """Transcribe a condition variant using Whisper."""
        self._load_pipeline()

        # Load audio from generated variant file
        # The variant_id encodes the path: clip_id__condition_id.wav
        # For now, construct the path from variant info
        audio_path = self._find_audio(variant)
        if not audio_path:
            return Hypothesis(
                variant_id=variant.variant_id,
                model_id=self.model_id,
                hypothesis_text="",
                runner=self.name,
                runner_version=self.version,
                runtime_seconds=None,
            )

        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        audio_np = waveform.squeeze().numpy()

        start = time.time()
        result = self._pipeline(audio_np)
        elapsed = time.time() - start

        text = result.get("text", "") if isinstance(result, dict) else str(result)

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=text.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=round(elapsed, 3),
        )

    @staticmethod
    def _find_audio(variant: ConditionVariant) -> Path | None:
        """Find the generated audio file for a variant."""
        # Check standard location
        path = Path(f"data/generated/{variant.variant_id}.wav")
        if path.exists():
            return path
        return None
