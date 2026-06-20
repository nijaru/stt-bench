"""Parakeet TDT 1.1B runner.

Uses NeMo toolkit for inference.
Model: nvidia/parakeet-tdt-1.1b
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

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

        self._model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=self.model_id,
        )
        self._model.eval()

    def transcribe(
        self, variant: ConditionVariant, audio_dir: Path | None = None
    ) -> Hypothesis:
        """Transcribe a single audio file."""
        self._load()

        audio_path = self.find_audio(variant, audio_dir)

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        start = time.monotonic()

        # NeMo expects a list of file paths or numpy arrays
        transcription = self._model.transcribe(
            paths2audio_files=[str(audio_path)]
        )[0]

        elapsed = time.monotonic() - start

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=transcription.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=elapsed,
        )
