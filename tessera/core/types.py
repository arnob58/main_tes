"""Plain data types shared across subsystems.

SRP: this module only defines data shapes. No behaviour beyond trivial
helpers lives here, so every other module can import it without pulling in
inference, storage or UI code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    @staticmethod
    def system(text: str) -> "Message":
        return Message(Role.SYSTEM, text)

    @staticmethod
    def user(text: str) -> "Message":
        return Message(Role.USER, text)

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(Role.ASSISTANT, text)


@dataclass(frozen=True)
class GenerationResult:
    """Inference output plus the usage metadata the runtime actually reported."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # "stop" | "length" | "error" | backend-specific
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
        }


class Complexity(str, enum.Enum):
    INSTANT = "instant"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass(frozen=True)
class RouteDecision:
    complexity: Complexity
    needs_skills: bool
    raw: str
    parsed_cleanly: bool  # False when the parser had to fall back

    def label(self) -> str:
        return f"{self.complexity.value.capitalize()}|{'yes' if self.needs_skills else 'no'}"


@dataclass(frozen=True)
class Handoff:
    """Concise output passed from one logical agent to the next."""

    agent: str
    text: str


@dataclass(frozen=True)
class SkillRequest:
    skill: str
    arguments: dict[str, Any]
    purpose: str = ""   # the task objective, for generation context
    task_id: str = ""   # selects the task workspace


@dataclass(frozen=True)
class Observation:
    """Structured result of anything executed outside the model."""

    status: str  # "success" | "error" | "unavailable"
    source: str
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    artifacts: tuple[str, ...] = ()
    duration_ms: int = 0
    notes: tuple[str, ...] = ()  # what the engine did: "created via recipe", "installed pypdf"…

    def summary(self, limit: int = 1200) -> str:
        """Compact text form for injection into a prompt."""
        parts = [f"[{self.source}] status={self.status}"]
        if self.notes:
            parts.append("notes: " + "; ".join(self.notes))
        if self.result is not None:
            parts.append(f"result: {self.result!s}")
        if self.stdout:
            parts.append(f"stdout: {self.stdout}")
        if self.stderr:
            parts.append(f"stderr: {self.stderr}")
        if self.artifacts:
            parts.append(f"artifacts: {', '.join(self.artifacts)}")
        text = "\n".join(parts)
        return text if len(text) <= limit else text[: limit - 15] + "\n…[truncated]"
