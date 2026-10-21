"""Long-running, resumable, multi-step workflows.

SRP: break an objective into steps, run each step through the existing
reasoning pipeline, checkpoint after every step, and pause at consequential
steps until the user confirms. It contains no domain logic — "apply to jobs
on LinkedIn" is just steps that use generic skills (desktop_task,
find_files, read_pdf…).

Honesty rules
  * a step is `done` only if its pipeline completed; a consequential step
    additionally needs a *successful tool observation* — text alone is
    recorded as `unverified`, never as done
  * a workflow is `completed` only when every step is done or skipped
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..agents.pipeline import Pipeline
from ..context.checkpoint import CheckpointStore
from .events import NullEvents
from .interfaces import Approver, EventSink, ModelProvider
from .task_state import CancellationToken, TaskCancelled, TaskStatus, WorkingMemory
from .types import Complexity, Message, RouteDecision

PLAN_PROMPT = """Stage: workflowplan.
Break the objective into 2-10 concrete steps that can each be done with the available capabilities.
Reply with JSON only: {"steps": [{"goal": "...", "consequential": false}]}
Mark "consequential": true for steps that submit, send, buy, apply, delete or publish anything."""


@dataclass
class Step:
    id: int
    goal: str
    consequential: bool = False
    status: str = "pending"      # pending | running | done | unverified | failed | skipped | waiting_for_user
    result: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class Workflow:
    objective: str
    steps: list[Step]
    id: str = field(default_factory=lambda: "wf_" + uuid.uuid4().hex[:10])
    status: str = TaskStatus.CREATED.value
    approved_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Workflow":
        return Workflow(d["objective"], [Step(**s) for s in d["steps"]], d["id"], d["status"], d.get("approved_steps", []))

    def summary(self) -> str:
        lines = [f"Workflow {self.id}: {self.status}"]
        for s in self.steps:
            mark = {"done": "✓", "unverified": "?", "failed": "✗", "waiting_for_user": "⏸", "skipped": "–"}.get(s.status, "·")
            lines.append(f"  {mark} {s.id}. {s.goal}" + (f" — {s.result[:120]}" if s.result else ""))
        return "\n".join(lines)


def parse_steps(text: str) -> list[Step]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in plan")
    data = json.loads(m.group(0))
    steps = [Step(i + 1, str(s["goal"]).strip(), bool(s.get("consequential", False)))
             for i, s in enumerate(data.get("steps", [])) if str(s.get("goal", "")).strip()]
    if not steps:
        raise ValueError("plan has no steps")
    return steps[:10]


class WorkflowRunner:
    def __init__(self, model: ModelProvider, pipeline: Pipeline, checkpoints: CheckpointStore, approver: Approver,
                 *, events: EventSink | None = None, plan_tokens: int = 400) -> None:
        self._model = model
        self._pipeline = pipeline
        self._store = checkpoints
        self._approver = approver
        self._events = events or NullEvents()
        self._plan_tokens = plan_tokens

    # ------------------------------------------------------------ lifecycle
    def create(self, objective: str, steps: Sequence[Step] | None = None) -> Workflow:
        if steps is None:
            r = self._model.generate([Message.system(PLAN_PROMPT), Message.user(objective)],
                                     max_output_tokens=self._plan_tokens, temperature=0.2)
            steps = parse_steps(r.text)
        wf = Workflow(objective, list(steps))
        wf.status = TaskStatus.PLANNING.value
        self._save(wf)
        return wf

    def load(self, workflow_id: str) -> Workflow | None:
        d = self._store.load(f"workflow_{workflow_id}")
        return Workflow.from_dict(d["workflow"]) if d else None

    def pending(self) -> list[Workflow]:
        out = []
        for rec in self._store.list("workflow_"):
            wf = Workflow.from_dict(rec["workflow"])
            if wf.status not in (TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value):
                out.append(wf)
        return out

    def approve(self, wf: Workflow, step_id: int) -> None:
        if step_id not in wf.approved_steps:
            wf.approved_steps.append(step_id)
        for s in wf.steps:
            if s.id == step_id and s.status == "waiting_for_user":
                s.status = "pending"
        self._save(wf)

    def run(self, wf: Workflow, cancel: CancellationToken | None = None) -> Workflow:
        cancel = cancel or CancellationToken()
        wf.status = TaskStatus.RUNNING.value
        self._save(wf)
        try:
            for step in wf.steps:
                if step.status in ("done", "skipped", "unverified"):
                    continue  # resumed workflow: never redo finished work
                cancel.raise_if_cancelled()
                if step.consequential and step.id not in wf.approved_steps:
                    if not self._approver.approve("workflow_step", f"Step {step.id}: {step.goal}"):
                        step.status = "waiting_for_user"
                        wf.status = TaskStatus.WAITING_FOR_USER.value
                        self._events.emit("Workflow", f"Paused before step {step.id} — needs your confirmation")
                        self._save(wf)
                        return wf
                    wf.approved_steps.append(step.id)
                self._run_step(wf, step, cancel)
                self._save(wf)
                if step.status == "failed":
                    wf.status = TaskStatus.FAILED.value
                    self._save(wf)
                    return wf
            all_done = all(s.status in ("done", "skipped") for s in wf.steps)
            wf.status = TaskStatus.COMPLETED.value if all_done else TaskStatus.WAITING_FOR_USER.value
        except TaskCancelled:
            wf.status = TaskStatus.CANCELLED.value
        self._save(wf)
        return wf

    # ------------------------------------------------------------- internals
    def _run_step(self, wf: Workflow, step: Step, cancel: CancellationToken) -> None:
        step.status = "running"
        self._save(wf)
        self._events.emit("Workflow", f"Step {step.id}/{len(wf.steps)}: {step.goal}")
        done_before = "\n".join(f"- step {s.id} ({s.status}): {s.result[:200]}" for s in wf.steps if s.result) or "none"
        objective = (f"{step.goal}\n\n(This is step {step.id} of a larger objective: {wf.objective}\n"
                     f"Results of earlier steps:\n{done_before})")
        working = WorkingMemory(objective=objective, task_id=f"{wf.id}_s{step.id}")
        working.route = RouteDecision(Complexity.COMPLEX, True, "workflow", True)
        working.transition(TaskStatus.RUNNING)
        try:
            result = self._pipeline.run(working, (), cancel)
        except TaskCancelled:
            step.status = "pending"
            raise
        except Exception as exc:
            step.status, step.result = "failed", f"{type(exc).__name__}: {exc}"
            return
        step.result = result.text.strip()
        step.evidence = [o.summary(300) for o in working.observations]
        succeeded = any(o.status == "success" for o in working.observations)
        if step.consequential and not succeeded:
            step.status = "unverified"   # no tool evidence the action happened
        else:
            step.status = "done"

    def _save(self, wf: Workflow) -> None:
        self._store.save(f"workflow_{wf.id}", {"workflow": wf.to_dict()})
