"""VisionObserver: what is on this screenshot?

SRP: turn (image, question) into text using the *same loaded model* when it
really supports images. If it doesn't, say so explicitly — never pass an
image path to a text-only model and pretend it can see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VisionObservation:
    status: str             # "ok" | "unsupported" | "error"
    text: str
    image: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class VisionObserver(Protocol):
    @property
    def supported(self) -> bool: ...

    def observe(self, image_path: str | Path, question: str, *, max_output_tokens: int = 300) -> VisionObservation: ...


class ModelVision:
    """Adapter over any backend exposing describe_image() + vision_status()."""

    def __init__(self, backend, system: str = "") -> None:
        self._backend = backend
        self._system = system
        ok, reason = backend.vision_status()
        self._ok, self.reason = ok, reason

    @property
    def supported(self) -> bool:
        return self._ok

    def observe(self, image_path, question, *, max_output_tokens=300) -> VisionObservation:
        if not self._ok:
            return VisionObservation("unsupported", f"Vision unavailable: {self.reason}", str(image_path))
        try:
            r = self._backend.describe_image(image_path, question, max_output_tokens=max_output_tokens,
                                             system=self._system)
        except Exception as exc:
            return VisionObservation("error", f"{type(exc).__name__}: {exc}", str(image_path))
        return VisionObservation("ok", r.text, str(image_path), r.input_tokens, r.output_tokens)


class UnsupportedVision:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def supported(self) -> bool:
        return False

    def observe(self, image_path, question, *, max_output_tokens=300) -> VisionObservation:
        return VisionObservation("unsupported", f"Vision unavailable: {self.reason}", str(image_path))
