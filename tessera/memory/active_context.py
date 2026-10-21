"""System 1: the sliding window of the most recent conversation messages.

SRP: hold the last N user/assistant messages and announce evictions. It
does not archive or embed anything itself — Phase 2's MemoryManager
subscribes to the eviction hook and does that. System prompts, observations
and agent handoffs never enter this window.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from ..core.interfaces import EvictionHook
from ..core.types import Message, Role


class ActiveContext:
    def __init__(self, limit: int = 5) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._window: deque[Message] = deque()
        self._hooks: list[EvictionHook] = []
        self._append_hooks: list[EvictionHook] = []

    def on_evict(self, hook: EvictionHook) -> None:
        self._hooks.append(hook)

    def on_append(self, hook: EvictionHook) -> None:
        self._append_hooks.append(hook)

    def append(self, message: Message) -> None:
        if message.role is Role.SYSTEM:
            raise ValueError("System prompts don't belong in the conversational window.")
        self._window.append(message)
        for hook in self._append_hooks:
            hook(message)
        while len(self._window) > self._limit:
            evicted = self._window.popleft()
            for hook in self._hooks:
                hook(evicted)

    def extend(self, messages: Iterable[Message]) -> None:
        for m in messages:
            self.append(m)

    def messages(self) -> list[Message]:
        return list(self._window)

    def __len__(self) -> int:
        return len(self._window)
