"""Parakeet TDT 1.1B runner.

Uses NeMo toolkit for inference.
Model: nvidia/parakeet-tdt-1.1b
"""

from __future__ import annotations

import time
from pathlib import Path

from ..manifest import ConditionVariant, Hypothesis
from . import BaseRunner, register_runner


@register_runner("parakeet")
class ParakeetRunner(BaseRunner):
    """Parakeet TDT model runner using NeMo."""

    name = "parakeet_nemo"
    version = "0.1.0"

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-1.1b",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, **kwargs)
        self._model = None

    def _load(self):
        """Load the Parakeet TDT model via NeMo."""
        if self._model is not None:
            return

        import nemo.collections.asr as nemo_asr

        device = self._resolve_device()
        self._model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=self.model_id,
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._device = device

    def transcribe(self, variant: ConditionVariant, audio_dir: Path | None = None) -> Hypothesis:
        """Transcribe a single audio file."""
        self._load()

        audio_path = self.find_audio(variant, audio_dir)

        start = time.monotonic()

        result = self._model.transcribe(audio=[str(audio_path)], batch_size=1, verbose=False)
        transcription = result[0] if isinstance(result, list) else result
        if not isinstance(transcription, str):
            transcription = getattr(transcription, "text", str(transcription))

        elapsed = time.monotonic() - start

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=transcription.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=elapsed,
        )
