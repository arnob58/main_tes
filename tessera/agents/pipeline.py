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
                    if working.skill_calls < self._cfg.max_skill_calls:
                        working.observations.append(self._execute(working, action))
                        continue  # develop again with the observation
                    self._events.emit("Skills", "Skill-call limit reached; continuing without it.")
                else:
                    draft = text
                    working.revision_feedback = ""

            elif name == EVALUATE:
                verdict = parse_verdict(text)
                if not verdict.passed and working.revisions < self._cfg.max_revisions and DEVELOP in stages:
                    working.revisions += 1
                    working.revision_feedback = verdict.feedback
                    self._events.emit("Evaluate", f"Revision {working.revisions} requested.")
                    i = stages.index(DEVELOP)
                    continue

            elif name == RESPOND:
                return PipelineResult(text, working)

            i += 1

        raise RuntimeError("pipeline ended without a respond stage")  # guarded in __init__

    def _execute(self, working: WorkingMemory, action) -> Observation:
        working.skill_calls += 1
        if action.request is None:
            self._events.emit("Skills", f"Malformed action: {action.error}")
            return Observation(status="error", source="action-parser", stderr=action.error)
        req = replace(action.request, purpose=working.objective, task_id=working.task_id)
        self._events.emit("Skills", f"Requesting '{req.skill}'")
        working.transition(TaskStatus.WAITING_FOR_TOOL)
        try:
            obs = self._bridge.execute(req)
        except Exception as exc:  # the bridge must never crash reasoning
            obs = Observation(status="error", source=req.skill, stderr=f"{type(exc).__name__}: {exc}")
        finally:
            working.transition(TaskStatus.RUNNING)
        self._events.emit("Skills", f"'{req.skill}' → {obs.status}")
        return obs
