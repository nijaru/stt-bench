"""Model runners for STT-Bench.

Each runner implements the RunnerProtocol: load model, transcribe a condition variant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..manifest import ConditionVariant, Hypothesis


class RunnerProtocol(ABC):
    """Base class for model runners."""

    name: str = "unknown"
    version: str = "0.1.0"
    model_id: str = ""

    @abstractmethod
    def transcribe(self, variant: ConditionVariant) -> Hypothesis:
        """Transcribe a condition variant and return a hypothesis."""
        ...


# Registry of available runners
_RUNNERS: dict[str, type[RunnerProtocol]] = {}


def register_runner(model_prefix: str):
    """Decorator to register a runner for a model prefix."""

    def decorator(cls: type[RunnerProtocol]):
        _RUNNERS[model_prefix] = cls
        return cls

    return decorator


def get_runner(model_id: str, device: str = "auto") -> RunnerProtocol:
    """Get a runner instance for the given model ID.

    Matches model_id against registered prefixes (case-insensitive).
    Checks both startswith and contains for flexibility.
    """
    model_lower = model_id.lower()
    for prefix, cls in _RUNNERS.items():
        prefix_lower = prefix.lower()
        # Match: model starts with prefix, or model contains prefix after /
        if model_lower.startswith(prefix_lower) or "/" + prefix_lower in model_lower:
            return cls(model_id=model_id, device=device)

    # Fallback: try whisper runner for any HF model
    from .whisper_runner import WhisperRunner

    return WhisperRunner(model_id=model_id, device=device)


class BaseRunner(RunnerProtocol):
    """Base class with common runner functionality."""

    def __init__(self, model_id: str = "", device: str = "auto", **kwargs):
        self.model_id = model_id
        self.device = device

    def _resolve_device(self) -> str:
        """Resolve the device to use for inference."""
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device


# Import runner modules to trigger registration
from . import cohere_runner, parakeet_runner, qwen3_asr_runner, whisper_runner  # noqa: E402,F401

__all__ = ["RunnerProtocol", "BaseRunner", "get_runner", "register_runner"]
