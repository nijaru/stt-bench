"""Qwen3-ASR runner.

Uses qwen-asr library (Qwen3ASRModel).
Model: Qwen/Qwen3-ASR-1.7B
"""

from __future__ import annotations

import time
from pathlib import Path

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

        import torch
        from qwen_asr import Qwen3ASRModel

        device = self._resolve_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        load_kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "cuda"

        self._model = Qwen3ASRModel.from_pretrained(self.model_id, **load_kwargs)
        if device != "cuda":
            self._model.model = self._model.model.to(device)
        self._device = device

    def transcribe(self, variant: ConditionVariant, audio_dir: Path | None = None) -> Hypothesis:
        """Transcribe a single audio file."""
        self._load()

        audio_path = self.find_audio(variant, audio_dir)

        start = time.monotonic()

        # Qwen3-ASR expects file path, not audio array
        result = self._model.transcribe(str(audio_path))

        # Result is list of ASRTranscription objects
        if isinstance(result, list) and len(result) > 0:
            transcription = result[0].text
        else:
            transcription = str(result)

        elapsed = time.monotonic() - start

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=transcription.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=elapsed,
        )
