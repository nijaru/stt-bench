"""Parakeet TDT 1.1B runner.

Uses AutoModelForTDT + AutoProcessor from transformers.
Model: nvidia/parakeet-tdt-1.1b-v2
"""

from __future__ import annotations

import time
from pathlib import Path

import soundfile as sf

from ..manifest import ConditionVariant, Hypothesis
from . import register_runner, BaseRunner


@register_runner("parakeet")
class ParakeetRunner(BaseRunner):
    """Parakeet TDT model runner using transformers."""

    name = "parakeet"
    version = "0.1.0"

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-1.1b-v2",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, **kwargs)
        self._model = None

    def _load(self):
        """Load the Parakeet TDT model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        device = self._resolve_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForTDT.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        self._device = device
        self._dtype = dtype

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

        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device, dtype=self._dtype) if v.is_floating_point() else v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs)

        transcription = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        elapsed = time.monotonic() - start

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=transcription.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=elapsed,
        )
