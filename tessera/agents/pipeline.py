"""Sequential multi-agent reasoning over one shared model.

SRP: decide which stage runs next and feed it the right material. It does
not route, persist memory, or execute skills itself — skills go through the
injected ToolBridge.

Control flow (all loops bounded):
  stages come from config, e.g. interpret → develop → evaluate → respond
  develop emits ACTION  → bridge.execute → observation → develop again
  evaluate says REVISE  → develop again with the feedback
  pass budget nearly spent → jump straight to respond (an answer is always produced)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ..config import Config
from ..core.events import NullEvents
from ..core.interfaces import EventSink, ToolBridge
from ..core.task_state import CancellationToken, TaskStatus, WorkingMemory
from ..core.types import Handoff, Message, Observation
from .agent import Agent, AgentInput
from .parsing import parse_action, parse_verdict

import re
from pathlib import Path


def _files_block(saved: dict[str, str], cap: int = 3500) -> str:
    if not saved:
        return ""
    parts = []
    for path, text in saved.items():
        body = text if len(text) <= cap else text[:cap] + "\n…[file continues]"
        parts.append(f"FILE {path}:\n{body}")
    return "\n\n".join(parts)


def _fallback_reply(working) -> str:
    """Deterministic summary used when the model's final reply is empty."""
    lines = []
    for path in working.saved_files:
        lines.append(f"Saved: {path}")
    errors = [o for o in working.observations if o.status != "success" and not o.source.startswith("file:")]
    if errors and not working.saved_files:
        lines.append(f"I couldn't finish this. Last problem: {(errors[-1].stderr or str(errors[-1].result))[:300]}")
    return "\n".join(lines) or "I wasn't able to produce an answer for that. Could you rephrase or simplify the request?"


def _error_kind(text: str) -> str:
    """Normalise an error so reworded or renumbered versions of the same failure compare equal."""
    t = text.lower()
    t = re.sub(r"(['\"`]).*?\1", "x", t)          # quoted names/snippets
    t = re.sub(r"[a-z]:[\\/]\S+|/\S+", "path", t)     # paths
    t = re.sub(r"\d+", "n", t)                       # line numbers, counts
    t = re.sub(r"\s+", " ", t)
    return t[:120]


def _call_key(req) -> str:
    import hashlib
    import json as _json

    blob = _json.dumps([req.skill, req.arguments], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)
UNSAVED_CODE_FEEDBACK = (
    "You wrote code in your reply but did not save it. The user asked for a file. "
    "Use an ACTION to write it, with the complete file in a fenced code block right after the ACTION line."
)

RESPOND = "respond"
DEVELOP = "develop"
EVALUATE = "evaluate"
INTERPRET = "interpret"


@dataclass
class PipelineResult:
    text: str
    working: WorkingMemory


class Pipeline:
    def __init__(
        self,
        agents: Mapping[str, Agent],
        cfg: Config,
        bridge: ToolBridge,
        events: EventSink | None = None,
    ) -> None:
        self._agents = dict(agents)
        self._cfg = cfg
        self._bridge = bridge
        self._events = events or NullEvents()
        for complexity, stages in cfg.pipelines.items():
            missing = [s for s in stages if s not in self._agents]
            if missing:
                raise ValueError(f"Pipeline '{complexity}' references unknown agents: {missing}")
            if not stages or stages[-1] != RESPOND:
                raise ValueError(f"Pipeline '{complexity}' must end with '{RESPOND}'.")

    def run(
        self,
        working: WorkingMemory,
        history: Sequence[Message],
        cancel: CancellationToken,
    ) -> PipelineResult:
        assert working.route is not None, "route before running the pipeline"
        stages = self._cfg.pipelines[working.route.complexity.value]
        capabilities = self._bridge.describe_capabilities() if working.route.needs_skills else ""
        draft = ""
        i = 0

        while i < len(stages):
            cancel.raise_if_cancelled()
            name = stages[i]

            # Always leave one pass for the final answer.
            if name != RESPOND and working.agent_passes >= self._cfg.max_agent_iterations - 1:
                self._events.emit("Reasoning", "Pass budget reached; moving to the final answer.")
                i = len(stages) - 1
                continue

            self._events.emit("Agent", f"{name}…")
            inp = AgentInput(
                request=working.objective,
                history=history,
                analysis=working.latest(INTERPRET),
                draft=draft if name != INTERPRET else "",
                evaluation_feedback=working.revision_feedback,
                observations=tuple(working.observations),
                capabilities=capabilities,
                memory=working.memory_context,
                saved_files=_files_block(working.saved_files),
            )
            cancel.raise_if_cancelled()  # last safe point before a slow model call
            result = self._agents[name].run(inp)
            working.agent_passes += 1
            working.record_generation(result)
            text = result.text
            working.handoffs.append(Handoff(name, text))

            if name == DEVELOP:
                action = parse_action(text)
                if action is not None:
                    draft = action.preamble or draft
                    stalled = working.stalled_calls >= self._cfg.max_stalled_calls
                    if working.skill_calls < self._cfg.max_skill_calls and not stalled:
                        working.observations.append(self._execute(working, action))
                        continue  # develop again with the observation
                    if stalled:
                        self._events.emit("Skills", f"Stopping tools: {working.stalled_calls} attempts in a row without "
                                                    f"progress (last error: {working.last_error[:80]})")
                    else:
                        self._events.emit("Skills", "Skill-call limit reached; continuing without it.")
                else:
                    draft = text
                    working.revision_feedback = ""
                    if self._should_nudge(working, text):
                        working.save_nudges += 1
                        working.revision_feedback = UNSAVED_CODE_FEEDBACK
                        self._events.emit("Reasoning", "Code was written but not saved; asking for an ACTION.")
                        continue  # develop again

            elif name == EVALUATE:
                verdict = parse_verdict(text)
                if not verdict.passed and working.revisions < self._cfg.max_revisions and DEVELOP in stages:
                    working.revisions += 1
                    working.revision_feedback = verdict.feedback
                    self._events.emit("Evaluate", f"Revision {working.revisions} requested.")
                    i = stages.index(DEVELOP)
                    continue

            elif name == RESPOND:
                if not text.strip():
                    text = _fallback_reply(working)
                    self._events.emit("Reasoning", "The model gave an empty reply; summarising what happened.")
                return PipelineResult(text, working)

            i += 1

        raise RuntimeError("pipeline ended without a respond stage")  # guarded in __init__

    def _should_nudge(self, working: WorkingMemory, text: str) -> bool:
        """Develop printed a sizeable code block for a tool-needing task but saved nothing."""
        if not working.route or not working.route.needs_skills:
            return False
        if working.save_nudges >= self._cfg.max_save_nudges:
            return False
        if any(o.status == "success" and o.artifacts for o in working.observations):
            return False
        block = _CODE_BLOCK.search(text)
        return bool(block and block.group(1).count("\n") >= 8)

    def _execute(self, working: WorkingMemory, action) -> Observation:
        working.skill_calls += 1
        if action.request is None:
            self._events.emit("Skills", f"Malformed action: {action.error}")
            return Observation(status="error", source="action-parser", stderr=action.error)
        req = replace(action.request, purpose=working.objective, task_id=working.task_id)
        key = _call_key(req)
        target = Path(str(req.arguments.get("path") or req.arguments.get("folder") or "")).name
        already = next((p for p in working.saved_files if target and Path(p).name == target), None)
        if already:
            working.stalled_calls += 1
            self._events.emit("Skills", f"'{target}' is already saved and checked — not rewriting it")
            return Observation("error", req.skill, stderr=f"{already} is already saved and passed its checks. Do not "
                               "write it again. Create any OTHER files that are still missing; if none, reply with a "
                               "one-line summary and no ACTION.")
        if key in working.failed_calls:
            self._events.emit("Skills", f"Blocked an exact repeat of a failed '{req.skill}' call")
            return Observation("error", req.skill, stderr="This exact call already failed. Change the input to fix "
                                                         "the problem described earlier; do not repeat it unchanged.")
        self._events.emit("Skills", f"Requesting '{req.skill}'")
        working.transition(TaskStatus.WAITING_FOR_TOOL)
        try:
            obs = self._bridge.execute(req)
        except Exception as exc:  # the bridge must never crash reasoning
            obs = Observation(status="error", source=req.skill, stderr=f"{type(exc).__name__}: {exc}")
        finally:
            working.transition(TaskStatus.RUNNING)
        self._events.emit("Skills", f"'{req.skill}' → {obs.status}")
        if obs.status == "error":
            working.failed_calls.add(key)
            sig = _error_kind(obs.stderr or str(obs.result))
            # A different *kind* of error means the last fix changed something — that's progress.
            # Same kind again, or the same tool failing 4 times running, is not.
            working.tool_streak = working.tool_streak + 1 if working.last_tool == req.skill else 1
            working.last_tool = req.skill
            same = sig == working.last_error
            working.stalled_calls = working.stalled_calls + 1 if same else 1
            if working.tool_streak >= 4:
                working.stalled_calls = max(working.stalled_calls, self._cfg.max_stalled_calls)
            working.last_error = sig
        else:
            working.stalled_calls = 0
            working.tool_streak = 0
            working.last_error = ""
        if obs.status == "success":
            for a in obs.artifacts:
                p = Path(a)
                if p.is_file() and p.stat().st_size < 300_000 and not p.name.endswith(".partial"):
                    working.saved_files[str(p)] = p.read_text(encoding="utf-8", errors="replace")
            if obs.artifacts:
                obs = replace(obs, notes=obs.notes + ("SAVED AND CHECKED — do not write this file again. If the request "
                                                      "needs no other files, reply with a one-line summary, no ACTION.",))
        return obs
