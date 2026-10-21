"""Approver implementations. The console supplies an interactive one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StaticApprover:
    """Always answers the same way; records what it was asked (tests, headless runs)."""

    answer: bool = False
    asked: list[tuple[str, str]] = field(default_factory=list)

    def approve(self, action: str, detail: str) -> bool:
        self.asked.append((action, detail))
        return self.answer


@dataclass
class PolicyApprover:
    """Auto-approves listed action kinds, delegates the rest."""

    auto: frozenset[str]
    fallback: Callable[[str, str], bool]

    def approve(self, action: str, detail: str) -> bool:
        return action in self.auto or self.fallback(action, detail)
