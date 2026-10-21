"""SkillEngine — the real ToolBridge.

SRP: run the resolve → build → check → execute → observe → recover →
promote loop for one request. Each step is delegated:

  resolver     L1 skill / L2 recipe / L3 generate
  generator    model-written or model-repaired code (L3 and repairs)
  policy       static risk scan → Approver
  runner       child process execution + observation
  recovery     mechanical fix vs. escalation
  dependencies approved package installs

The engine keeps its own per-request working state (SkillJob). Only the
final Observation goes back to the reasoning pipeline.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..core.events import NullEvents
from ..core.interfaces import Approver, EventSink
from ..core.types import Observation, SkillRequest
from ..execution.artifacts import WorkspaceManager
from ..execution.process_runner import ProcessRunner, RunResult
from .contract import ContractError, normalise_name, parse_skill_source
from .dependencies import DependencyManager
from .generator import GenerationError, SkillGenerator
from .policy import risky_operations
from .recovery import Action, SkillRecovery
from .registry import SkillRegistry
from .resolver import SkillResolver

log = logging.getLogger("tessera.skills.engine")


@dataclass
class SkillJob:
    objective: str
    requested: str
    current_skill: str = ""
    level: int = 0
    attempt: int = 0
    dependencies: list[str] = field(default_factory=list)
    status: str = "running"
    artifacts: list[str] = field(default_factory=list)
    last_error: str | None = None
    notes: list[str] = field(default_factory=list)


class SkillEngine:
    def __init__(
        self,
        registry: SkillRegistry,
        resolver: SkillResolver,
        runner: ProcessRunner,
        workspaces: WorkspaceManager,
        promote_dir: Path,
        approver: Approver,
        dependencies: DependencyManager,
        generator: SkillGenerator | None = None,
        *,
        events: EventSink | None = None,
        max_retries: int = 3,
        max_listed: int = 25,
        on_job=None,
    ) -> None:
        self._on_job = on_job  # Callable[[SkillJob], None] — e.g. metrics
        self.registry = registry
        self._resolver = resolver
        self._runner = runner
        self._workspaces = workspaces
        self._promote_dir = promote_dir
        self._approver = approver
        self._deps = dependencies
        self._generator = generator
        self._events = events or NullEvents()
        self._max_retries = max_retries
        self._max_listed = max_listed
        self.jobs: list[SkillJob] = []

    # ------------------------------------------------------------ ToolBridge
    def describe_capabilities(self) -> str:
        self.registry.refresh()
        specs = self.registry.all()
        lines = [s.signature() for s in specs[: self._max_listed]]
        if len(specs) > self._max_listed:
            lines.append(f"…and {len(specs) - self._max_listed} more")
        head = "Installed skills:\n" + ("\n".join(lines) if lines else "(none yet)")
        if self._generator is not None:
            head += (
                "\nIf none fits, request a new capability with a descriptive snake_case name "
                "(e.g. read_pdf, find_files) and the arguments it needs; it will be written, "
                "tested and kept if it works."
            )
        return head

    def execute(self, request: SkillRequest) -> Observation:
        job = SkillJob(objective=request.purpose, requested=request.skill)
        self.jobs.append(job)
        try:
            obs = self._execute(request, job)
        except Exception as exc:  # engine bugs become observations, never crashes
            log.exception("skill engine failure")
            obs = Observation("error", request.skill, stderr=f"engine error: {type(exc).__name__}: {exc}")
        job.status = obs.status
        if self._on_job is not None:
            try:
                self._on_job(job)
            except Exception:
                log.exception("on_job hook failed")
        return obs

    # ------------------------------------------------------------- internals
    def _execute(self, request: SkillRequest, job: SkillJob) -> Observation:
        self.registry.refresh()
        name = normalise_name(request.skill)
        native = self.registry.native(name)
        if native is not None:
            job.current_skill, job.level = name, 1
            self._events.emit("Skills", f"Using '{name}'")
            return native.invoke(request.arguments, task_id=request.task_id, purpose=request.purpose)

        resolution = self._resolver.resolve(request)
        job.level = resolution.level
        workspace = self._workspaces.for_task(request.task_id)
        source: str | None = None
        repairable = True

        if resolution.level == 1:
            spec = resolution.skill
            job.current_skill = spec.name
            path = spec.path
            repairable = spec.origin != "builtin"
            if resolution.reason != "exact name":
                job.notes.append(f"used existing skill '{spec.name}' ({resolution.reason})")
            self._events.emit("Skills", f"Using '{spec.name}'")
            if repairable:  # repairs happen on a workspace copy, never the original
                source = path.read_text(encoding="utf-8")
        elif resolution.level == 2:
            recipe = resolution.recipe
            job.current_skill = recipe.name
            self._events.emit("Skills", f"No skill for '{name}'; building from recipe '{recipe.name}'")
            source = recipe.build(request)
            job.notes.append(f"built from recipe '{recipe.name}' (no model call)")
        else:
            if self._generator is None:
                return Observation("unavailable", name, result="No skill matches and generation is disabled.")
            self._events.emit("Skills", f"'{name}' missing; generating…")
            try:
                gen = self._generator.generate(request, name)
            except GenerationError as exc:
                return Observation("error", name, stderr=str(exc), notes=("generation failed; nothing was executed",))
            source = gen.source
            job.current_skill = gen.spec.name
            job.notes.append(f"generated new skill '{gen.spec.name}' (attempt {gen.attempts})")
            self._events.emit("Skills", f"Generated {gen.spec.name}.py")

        arguments = dict(request.arguments)
        checked_sources: set[str] = set()
        timeout_retried = False
        timeout = None
        recovery = SkillRecovery()
        last: RunResult | None = None

        for attempt in range(1, self._max_retries + 2):
            job.attempt = attempt
            if source is not None:
                path = workspace / f"{job.current_skill}.py"
                path.write_text(source, encoding="utf-8")
                if source not in checked_sources:
                    checked_sources.add(source)
                    risky = risky_operations(source)
                    if risky and not self._approver.approve("run_risky_code", f"{job.current_skill}: {', '.join(risky)}"):
                        return self._result_obs(job, None, "denied", f"not run: uses {', '.join(risky)} and was not approved")

            self._events.emit("Execution", f"Running {job.current_skill} (attempt {attempt})")
            last = self._runner.run(path, arguments, workspace, timeout_s=timeout)
            job.artifacts.extend(a for a in last.artifacts if a not in job.artifacts)
            if last.ok:
                if resolution.level > 1 or source is not None and job.attempt > 1:
                    self._promote(job, source)
                return self._result_obs(job, last, "success")

            job.last_error = f"{last.error_type}: {last.message}"
            self._events.emit("Execution", f"Failed: {job.last_error[:120]}")
            plan = recovery.plan(last, arguments, timeout_retried=timeout_retried)

            if plan.action is Action.INSTALL:
                outcome = self._deps.ensure(plan.detail)
                job.dependencies.append(outcome.package)
                job.notes.append(f"dependency {outcome.package}: {outcome.detail}")
                self._events.emit("Dependencies", f"{outcome.package}: {outcome.detail}")
                if not outcome.installed:
                    break
            elif plan.action is Action.RETRY_LONGER:
                timeout_retried = True
                timeout = self._runner.timeout_s * 3
                job.notes.append("timed out; retrying once with a longer limit")
            elif plan.action is Action.FIX_ARGS:
                arguments = plan.fixed_args or arguments
                job.notes.append(plan.detail)
            elif plan.action is Action.REPAIR and self._generator is not None and repairable and source is not None:
                self._events.emit("Skills", f"Asking the model to repair {job.current_skill}")
                try:
                    fix = self._generator.repair(source, last.traceback or job.last_error, job.current_skill, arguments)
                except GenerationError as exc:
                    job.notes.append(f"repair failed: {exc}")
                    break
                source = fix.source
                job.notes.append(f"repaired by the model after {last.error_type}")
            else:
                if plan.action is Action.GIVE_UP:
                    job.notes.append(plan.detail)
                break

        return self._result_obs(job, last, "error")

    def _promote(self, job: SkillJob, source: str | None) -> None:
        if source is None:
            return
        try:
            spec = parse_skill_source(source, default_name=job.current_skill)
        except ContractError:
            return
        self._promote_dir.mkdir(parents=True, exist_ok=True)
        target = self._promote_dir / f"{spec.name}.py"
        if target.exists():
            if target.read_text(encoding="utf-8") == source:
                return
            backup = target.with_suffix(f".py.bak{len(list(self._promote_dir.glob(spec.name + '.py.bak*')))}")
            shutil.copy2(target, backup)
            job.notes.append(f"previous version backed up to {backup.name}")
        target.write_text(source, encoding="utf-8")
        self.registry.refresh()
        job.notes.append(f"registered '{spec.name}' for reuse (validated by this run only)")
        self._events.emit("Skills", f"Registered {spec.name}")

    def _result_obs(self, job: SkillJob, run: RunResult | None, status: str, extra: str = "") -> Observation:
        notes = tuple(job.notes + ([extra] if extra else []))
        if run is None:
            return Observation(status, job.current_skill or job.requested, notes=notes)
        stderr = run.stderr
        if not run.ok:
            stderr = (f"{run.error_type}: {run.message}\n{run.traceback}" if run.traceback else f"{run.error_type}: {run.message}\n{stderr}").strip()
        return Observation(
            status=status,
            source=job.current_skill,
            result=run.result if run.ok else None,
            stdout=run.stdout.strip(),
            stderr=stderr,
            artifacts=run.artifacts,
            duration_ms=run.duration_ms,
            notes=notes,
        )
