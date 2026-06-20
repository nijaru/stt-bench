"""Cohere Transcribe runner.

Uses CohereAsrForConditionalGeneration + AutoProcessor from transformers.
Model: CohereLabs/cohere-transcribe-03-2026
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf

from ..manifest import ConditionVariant, Hypothesis
from . import BaseRunner, register_runner


@register_runner("cohere")
class CohereRunner(BaseRunner):
    """Cohere Transcribe model runner using transformers."""

    name = "cohere"
    version = "0.1.0"

    def __init__(
        self,
        model_id: str = "CohereLabs/cohere-transcribe-03-2026",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_id=model_id, device=device, **kwargs)
        self._model = None
        self._processor = None

    def _load(self):
        """Load the Cohere ASR model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        device = self._resolve_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self._processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            token=token,
        )
        if device != "cuda":
            self._model = self._model.to(device)

        self._model.eval()
        self._device = device
        self._dtype = dtype

    def transcribe(self, variant: ConditionVariant, audio_dir: Path | None = None) -> Hypothesis:
        """Transcribe a single audio file."""
        import torch

        self._load()

        audio_path = self.find_audio(variant, audio_dir)

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
        inputs = {
            k: v.to(self._model.device, dtype=self._dtype)
            if v.is_floating_point()
            else v.to(self._model.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs)

        transcription = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = time.monotonic() - start

        return Hypothesis(
            variant_id=variant.variant_id,
            model_id=self.model_id,
            hypothesis_text=transcription.strip(),
            runner=self.name,
            runner_version=self.version,
            runtime_seconds=elapsed,
        )
