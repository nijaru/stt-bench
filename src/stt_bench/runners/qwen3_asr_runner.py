"""Qwen3-ASR runner.

Uses qwen-asr library (Qwen3ASRModel).
Model: Qwen/Qwen3-ASR-1.7B
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf

from ..manifest import ConditionVariant, Hypothesis
from . import BaseRunner, register_runner


@register_runner("qwen3-asr")
class Qwen3ASRRunner(BaseRunner):
    """Qwen3-ASR model runner using qwen-asr library."""

    name = "qwen3-asr"
    version = "0.1.0"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, **kwargs)
        self._model = None

    def _load(self):
        """Load the Qwen3-ASR model."""
        if self._model is not None:
            return

        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(self.model_id)
        self._model.eval()

        device = self._resolve_device()
        if device != "cpu":
            self._model = self._model.to(device)
        self._device = device

    def transcribe(self, variant: ConditionVariant) -> Hypothesis:
        """Transcribe a single audio file."""
        import torch

        self._load()

        audio_path = Path(variant.source_uri)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        start = time.monotonic()

        with torch.no_grad():
            result = self._model.transcribe(audio, sampling_rate=16000)

        transcription = result.get("text", "") if isinstance(result, dict) else str(result)
        elapsed = time.monotonic() - start

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=transcription.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=elapsed,
        )
