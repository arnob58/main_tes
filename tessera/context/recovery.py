"""Context recovery for long code generation.

SRP: drive a generation across as many *fresh* inference contexts as it
needs, with the artifact on disk as the single source of truth.

Per round:
  A  bound output so prompt + output stays under the threshold (90%)
  B  generation reaches that bound → write the exact text to disk
  C  analyse with `ast`: complete prefix, exact incomplete suffix, interfaces
  D  checkpoint objective / completed / pending / interfaces / errors
  E  reset the model's KV state (if the backend supports it) and build a new
     prompt from the checkpoint — never from the previous transcript
  F  continue; merge without regenerating completed code
  G  validate the integrated artifact (ast + compile)
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..core.events import NullEvents
from ..core.interfaces import EventSink, ModelProvider
from ..core.types import Message
from .checkpoint import CheckpointStore
from .continuation import WRITE_PROMPT, build_continuation_messages, merge_continuation, strip_fences
from .source_analyzer import analyze
from .tracker import ContextTracker

log = logging.getLogger("tessera.context")


class RecoveryFailed(RuntimeError):
    pass


@dataclass
class RoundRecord:
    prompt_tokens: int
    output_budget: int
    output_tokens: int
    finish_reason: str
    usage_ratio: float
    fresh_context: bool


@dataclass
class LongGenResult:
    path: Path
    source: str
    valid: bool
    error: str
    rounds: list[RoundRecord] = field(default_factory=list)
    recoveries: int = 0


class LongCodeGenerator:
    def __init__(
        self,
        model: ModelProvider,
        tracker: ContextTracker,
        checkpoints: CheckpointStore,
        *,
        events: EventSink | None = None,
        max_rounds: int = 12,
        min_output_tokens: int = 64,
        temperature: float = 0.2,
        prompt_share: float = 0.5,
    ) -> None:
        self._model = model
        self._tracker = tracker
        self._checkpoints = checkpoints
        self._events = events or NullEvents()
        self._max_rounds = max_rounds
        self._min_output = min_output_tokens
        self._temperature = temperature
        self._prompt_share = prompt_share  # share of the threshold budget a continuation prompt may use

    def generate(self, objective: str, path: Path, *, key: str, context: str = "", resume: bool = True) -> LongGenResult:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = self._checkpoints.load(key) if resume else None
        completed: list[str] = []
        source = ""
        recoveries = 0
        if saved and path.exists():
            source = path.read_text(encoding="utf-8")
            completed = list(saved.get("completed_steps", []))
            recoveries = int(saved.get("recoveries", 0))
            messages = self._continuation(objective, source, completed, suspect=True)
            self._events.emit("Context", f"Resuming {path.name} from checkpoint")
        else:
            user = f"TASK:\n{objective}" + (f"\n\nCONTEXT:\n{context}" if context else "")
            messages = [Message.system(WRITE_PROMPT), Message.user(user)]

        rounds: list[RoundRecord] = []
        fresh = bool(saved)
        for _ in range(self._max_rounds):
            usage = self._tracker.measure(messages)
            budget = usage.safe_output_budget
            if budget < self._min_output:
                raise RecoveryFailed(
                    f"prompt uses {usage.prompt_tokens}/{usage.window} tokens; no room to generate safely"
                )
            result = self._model.generate(messages, max_output_tokens=budget, temperature=self._temperature)
            rounds.append(RoundRecord(usage.prompt_tokens, budget, result.output_tokens, result.finish_reason,
                                      (usage.prompt_tokens + result.output_tokens) / usage.window, fresh))
            chunk = strip_fences(result.text)
            source = merge_continuation(source, chunk)
            path.write_text(source, encoding="utf-8")  # B: exact text on disk, always

            cut_off = result.finish_reason == "length"
            if not cut_off and _parses(source):
                break

            # ---- recovery event
            recoveries += 1
            state = analyze(source, last_unit_suspect=cut_off)
            completed = [f"{f.name}{f.signature}" for f in state.functions] + [f"class {c.name}" for c in state.classes]
            pending = "finish the incomplete section, then the rest of the module" if state.incomplete_suffix.strip() \
                else "write the remaining definitions"
            self._checkpoints.save(key, {
                "objective": objective,
                "artifact": str(path),
                "completed_steps": completed,
                "pending": pending,
                "interfaces": state.interfaces_text(),
                "incomplete_suffix": state.incomplete_suffix,
                "parse_error": state.parse_error,
                "recoveries": recoveries,
                "usage_at_trigger": rounds[-1].usage_ratio,
            })
            self._events.emit(
                "Context",
                f"{rounds[-1].usage_ratio:.0%} of context used — saved {path.name}, continuing in a fresh context",
            )
            reset = getattr(self._model, "reset_context", None)
            fresh = callable(reset)
            if fresh:
                reset()
            messages = self._continuation(objective, source, completed, suspect=cut_off, state=state)
        else:
            return self._finish(path, source, rounds, recoveries, key, "round limit reached before completion")

        return self._finish(path, source, rounds, recoveries, key, "")

    # ------------------------------------------------------------- helpers
    def _continuation(self, objective, source, completed, *, suspect, state=None):
        state = state or analyze(source, last_unit_suspect=suspect)
        return build_continuation_messages(
            objective, state,
            completed_steps=completed,
            next_operation=(
                "Continue the incomplete section from its last character, then write the remaining definitions."
                if state.incomplete_suffix.strip() else "Write the remaining definitions of the module."
            ),
            count=self._tracker.count,
            max_prompt_tokens=int(self._tracker.window * self._tracker.threshold * self._prompt_share),
        )

    def _finish(self, path, source, rounds, recoveries, key, error) -> LongGenResult:
        valid, err = validate_python(source, str(path))
        if valid and not error:
            self._checkpoints.discard(key)
        return LongGenResult(path, source, valid and not error, error or err, rounds, recoveries)


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def validate_python(source: str, filename: str = "<generated>") -> tuple[bool, str]:
    try:
        compile(source, filename, "exec")
        return True, ""
    except SyntaxError as exc:
        return False, f"{type(exc).__name__} at line {exc.lineno}: {exc.msg}"
