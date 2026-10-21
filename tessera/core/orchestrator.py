"""Coordinates one user turn: route → reason → remember → reply.

SRP: sequencing and lifecycle. The orchestrator owns no model, no prompts,
no storage format and no UI; it receives collaborators that satisfy small
interfaces and wires their outputs together.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..agents.pipeline import Pipeline
from ..core.events import NullEvents
from ..core.interfaces import EventSink, MemoryRetriever
from ..core.task_state import CancellationToken, TaskCancelled, TaskStatus, WorkingMemory
from ..core.types import Message, RouteDecision
from ..llm.router import Router
from ..memory.active_context import ActiveContext

log = logging.getLogger("tessera.orchestrator")


@dataclass(frozen=True)
class Reply:
    text: str
    status: TaskStatus
    route: RouteDecision | None
    model_calls: int
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    task: dict


class Orchestrator:
    def __init__(
        self,
        router: Router,
        pipeline: Pipeline,
        context: ActiveContext,
        events: EventSink | None = None,
        memory: MemoryRetriever | None = None,
        recall_k: int = 5,
        checkpoints=None,
        on_reply=None,
    ) -> None:
        self._on_reply = on_reply  # Callable[[Reply], None] — e.g. metrics
        self._checkpoints = checkpoints  # CheckpointStore | None — interrupted tasks stay inspectable
        self._memory = memory
        self._recall_k = recall_k
        self._router = router
        self._pipeline = pipeline
        self._context = context
        self._events = events or NullEvents()

    def _checkpoint(self, working: WorkingMemory) -> None:
        if self._checkpoints is not None:
            self._checkpoints.save(f"task_{working.task_id}", {"task": working.snapshot(), "error": working.error})

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def context(self) -> ActiveContext:
        return self._context

    def handle(self, user_text: str, cancel: CancellationToken | None = None) -> Reply:
        cancel = cancel or CancellationToken()
        working = WorkingMemory(objective=user_text.strip())
        history = self._context.messages()
        text = ""

        try:
            working.transition(TaskStatus.PLANNING)
            self._events.emit("Router", "Classifying…")
            working.route = self._router.route(working.objective, history)
            working.model_calls += 1  # router call (usage tracked by engine metrics later)
            note = "" if working.route.parsed_cleanly else " (recovered from malformed output)"
            self._events.emit("Router", working.route.label() + note)

            if self._memory is not None:
                working.memory_context = self._memory.recall_text(working.objective, self._recall_k)
                n = working.memory_context.count("\n[#")
                self._events.emit("Memory", f"{n} related memories" if n else "No related memories.")

            cancel.raise_if_cancelled()
            working.transition(TaskStatus.RUNNING)
            self._checkpoint(working)
            result = self._pipeline.run(working, history, cancel)
            text = result.text
            working.transition(TaskStatus.COMPLETED)

            # Only the real exchange enters conversational memory — never
            # handoffs, router output or observations.
            self._context.append(Message.user(working.objective))
            self._context.append(Message.assistant(text))

        except TaskCancelled:
            working.transition(TaskStatus.CANCELLED)
            text = "Cancelled."
            self._events.emit("Task", "Cancelled by user.")
        except Exception as exc:
            log.exception("task failed")
            working.error = f"{type(exc).__name__}: {exc}"
            if working.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                working.transition(TaskStatus.FAILED)
            text = f"Something went wrong: {working.error}"
            self._events.emit("Error", working.error)

        if working.status is TaskStatus.COMPLETED:
            if self._checkpoints is not None:
                self._checkpoints.discard(f"task_{working.task_id}")
        else:
            self._checkpoint(working)

        reply = Reply(
            text=text,
            status=working.status,
            route=working.route,
            model_calls=working.model_calls,
            input_tokens=working.input_tokens,
            output_tokens=working.output_tokens,
            elapsed_s=time.monotonic() - working.started_at,
            task=working.snapshot(),
        )
        if self._on_reply is not None:
            try:
                self._on_reply(reply)
            except Exception:
                log.exception("on_reply hook failed")
        return reply
