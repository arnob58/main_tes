"""Request-scoped working memory and task lifecycle.

Everything here is temporary: handoffs, observations, retry counters. It is
never written to long-term memory automatically. Phase 6 adds serialisation
for checkpoints; the `snapshot()` shape is already stable for that purpose.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .types import GenerationResult, Handoff, Observation, RouteDecision


class TaskStatus(str, enum.Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    WAITING_FOR_TOOL = "waiting_for_tool"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.CREATED: {TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED},
    TaskStatus.PLANNING: {TaskStatus.RUNNING, TaskStatus.WAITING_FOR_USER, TaskStatus.CANCELLED, TaskStatus.FAILED},
    TaskStatus.RUNNING: {
        TaskStatus.WAITING_FOR_USER, TaskStatus.WAITING_FOR_TOOL, TaskStatus.RECOVERING,
        TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.WAITING_FOR_USER: {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED},
    TaskStatus.WAITING_FOR_TOOL: {TaskStatus.RUNNING, TaskStatus.RECOVERING, TaskStatus.CANCELLED, TaskStatus.FAILED},
    TaskStatus.RECOVERING: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}


class InvalidTransition(RuntimeError):
    pass


class TaskCancelled(Exception):
    pass


class CancellationToken:
    """Thread-safe flag checked at every safe boundary (between model calls)."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TaskCancelled()


@dataclass
class WorkingMemory:
    objective: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.CREATED
    route: RouteDecision | None = None
    handoffs: list[Handoff] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    revision_feedback: str = ""
    memory_context: str = ""
    agent_passes: int = 0
    skill_calls: int = 0
    save_nudges: int = 0
    failed_calls: set[str] = field(default_factory=set)
    saved_files: dict[str, str] = field(default_factory=dict)   # path → content (capped), files made this turn
    stalled_calls: int = 0                                      # tool calls in a row without progress
    last_error: str = ""
    last_tool: str = ""
    tool_streak: int = 0                                        # consecutive failures of the same tool
    revisions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.monotonic)

    def transition(self, new: TaskStatus) -> None:
        if new not in _ALLOWED[self.status]:
            raise InvalidTransition(f"{self.status.value} → {new.value}")
        self.status = new

    def record_generation(self, result: GenerationResult) -> None:
        self.model_calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens

    def latest(self, agent: str) -> str:
        for h in reversed(self.handoffs):
            if h.agent == agent:
                return h.text
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "status": self.status.value,
            "route": self.route.label() if self.route else None,
            "handoffs": [(h.agent, h.text) for h in self.handoffs],
            "observations": [o.summary() for o in self.observations],
            "counters": {
                "agent_passes": self.agent_passes,
                "skill_calls": self.skill_calls,
                "revisions": self.revisions,
                "model_calls": self.model_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }
