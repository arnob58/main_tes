"""Progress events, decoupled from any particular UI.

Core code emits `(channel, message)` pairs; a console, desktop or web UI
subscribes. Nothing in core imports a UI module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("tessera.events")

Listener = Callable[[str, str], None]


@dataclass
class EventBus:
    _listeners: list[Listener]

    def __init__(self) -> None:
        self._listeners = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def emit(self, channel: str, message: str) -> None:
        log.debug("[%s] %s", channel, message)
        for listener in list(self._listeners):
            try:
                listener(channel, message)
            except Exception:  # a broken UI must never break a task
                log.exception("event listener failed")


class NullEvents:
    def emit(self, channel: str, message: str) -> None:  # pragma: no cover - trivial
        pass
