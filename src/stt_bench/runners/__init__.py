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

    Matches model_id against registered prefixes.
    """
    for prefix, cls in _RUNNERS.items():
        if model_id.startswith(prefix):
            return cls(model_id=model_id, device=device)

    # Fallback: try whisper runner for any HF model
    from .whisper_runner import WhisperRunner

    return WhisperRunner(model_id=model_id, device=device)


__all__ = ["RunnerProtocol", "get_runner", "register_runner"]
