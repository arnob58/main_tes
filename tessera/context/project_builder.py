"""Artifact-first generation of projects larger than one context window.

SRP: plan → generate one bounded module at a time → persist → validate →
record → integrate. Files on disk are authoritative; each module prompt
carries only the *interfaces* of its dependencies (via ast), never their
full text, and never the transcript of earlier modules.

A manifest (`.tessera_project.json`) makes the build resumable: finished
modules whose file hash still matches are skipped on re-run.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ..core.events import NullEvents
from ..core.interfaces import EventSink, ModelProvider
from ..core.types import Message
from .recovery import LongCodeGenerator, RecoveryFailed
from .source_analyzer import analyze

PLAN_PROMPT = """Stage: projectplan.
Plan a small Python project. Reply with JSON only, no prose:
{"modules": [{"path": "pkg/module.py", "purpose": "...", "interfaces": ["def name(arg: type) -> type", "class Name"], "depends_on": ["pkg/other.py"]}],
 "tests": [{"path": "tests/test_module.py", "purpose": "...", "depends_on": ["pkg/module.py"]}]}
Keep each module focused and small. Paths are relative, use forward slashes."""

MANIFEST = ".tessera_project.json"
_NAME = re.compile(r"(?:def|class)\s+([A-Za-z_]\w*)")


class PlanError(ValueError):
    pass


@dataclass
class ModuleSpec:
    path: str
    purpose: str
    interfaces: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    is_test: bool = False


@dataclass
class ModuleOutcome:
    path: str
    status: str             # done | invalid | failed | skipped
    detail: str = ""
    recoveries: int = 0
    missing_interfaces: list[str] = field(default_factory=list)


@dataclass
class ProjectReport:
    root: Path
    modules: list[ModuleOutcome]
    integration_ok: bool | None
    integration_output: str

    @property
    def ok(self) -> bool:
        return all(m.status in ("done", "skipped") for m in self.modules) and self.integration_ok is not False

    def summary(self) -> str:
        lines = [f"{m.path}: {m.status}" + (f" ({m.detail})" if m.detail else "") for m in self.modules]
        if self.integration_ok is not None:
            lines.append(f"integration: {'passed' if self.integration_ok else 'FAILED'}")
        return "\n".join(lines)


def parse_plan(text: str) -> list[ModuleSpec]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PlanError("no JSON object in plan")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError(f"invalid JSON: {exc.msg}") from exc
    specs = []
    for key, is_test in (("modules", False), ("tests", True)):
        for m in data.get(key, []) or []:
            path = str(m.get("path", ""))
            p = PurePosixPath(path)
            if not path.endswith(".py") or p.is_absolute() or ".." in p.parts:
                raise PlanError(f"unsafe or non-Python path: {path!r}")
            specs.append(ModuleSpec(path, str(m.get("purpose", "")), [str(i) for i in m.get("interfaces", []) or []],
                                    [str(d) for d in m.get("depends_on", []) or []], is_test))
    if not any(not s.is_test for s in specs):
        raise PlanError("plan has no modules")
    return specs


def build_order(specs: list[ModuleSpec]) -> list[ModuleSpec]:
    """Dependencies first; tests last; cycles fall back to plan order."""
    by_path = {s.path: s for s in specs}
    order: list[ModuleSpec] = []
    state: dict[str, int] = {}

    def visit(s: ModuleSpec) -> None:
        if state.get(s.path) == 2 or state.get(s.path) == 1:
            return
        state[s.path] = 1
        for d in s.depends_on:
            if d in by_path:
                visit(by_path[d])
        state[s.path] = 2
        order.append(s)

    for s in [s for s in specs if not s.is_test] + [s for s in specs if s.is_test]:
        visit(s)
    return [s for s in order if not s.is_test] + [s for s in order if s.is_test]


class ProjectBuilder:
    def __init__(self, model: ModelProvider, longgen: LongCodeGenerator, *, events: EventSink | None = None,
                 plan_tokens: int = 700, test_timeout_s: int = 300) -> None:
        self._model = model
        self._longgen = longgen
        self._events = events or NullEvents()
        self._plan_tokens = plan_tokens
        self._timeout = test_timeout_s

    def build(self, objective: str, root: Path, *, key: str) -> ProjectReport:
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / MANIFEST
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("objective") == objective and manifest.get("plan"):
            specs = [ModuleSpec(**m) for m in manifest["plan"]]
            self._events.emit("Project", f"Resuming build in {root}")
        else:
            specs = self._plan(objective)
            manifest = {"objective": objective, "plan": [s.__dict__ for s in specs], "modules": {}}
            self._save(manifest_path, manifest)
        self._events.emit("Project", f"{len(specs)} files planned")

        outcomes = []
        for spec in build_order(specs):
            target = root / spec.path
            record = manifest["modules"].get(spec.path, {})
            if record.get("status") == "done" and target.exists() and _sha(target) == record.get("sha"):
                outcomes.append(ModuleOutcome(spec.path, "skipped", "already built"))
                continue
            outcome = self._build_module(objective, root, spec, key)
            manifest["modules"][spec.path] = {"status": outcome.status, "sha": _sha(target) if target.exists() else ""}
            self._save(manifest_path, manifest)
            outcomes.append(outcome)
            self._events.emit("Project", f"{spec.path}: {outcome.status}")

        ok, output = self._integrate(root, any(s.is_test for s in specs))
        return ProjectReport(root, outcomes, ok, output)

    # ------------------------------------------------------------- steps
    def _plan(self, objective: str) -> list[ModuleSpec]:
        error = ""
        for _ in range(2):
            user = f"PROJECT:\n{objective}" + (f"\n\nYour previous plan was invalid: {error}" if error else "")
            r = self._model.generate([Message.system(PLAN_PROMPT), Message.user(user)],
                                     max_output_tokens=self._plan_tokens, temperature=0.2)
            try:
                return parse_plan(r.text)
            except PlanError as exc:
                error = str(exc)
        raise PlanError(f"could not obtain a valid plan: {error}")

    def _build_module(self, objective: str, root: Path, spec: ModuleSpec, key: str) -> ModuleOutcome:
        dep_interfaces = []
        for d in spec.depends_on:
            p = root / d
            if p.exists():
                text = analyze(p.read_text(encoding="utf-8")).interfaces_text()
                module = d[:-3].replace("/", ".")
                dep_interfaces.append(f"# {d}  (import as {module})\n{text}")
        context = (
            "Required interfaces:\n" + ("\n".join(spec.interfaces) or "(choose sensible ones)")
            + ("\n\nInterfaces of modules it depends on (already written):\n" + "\n\n".join(dep_interfaces) if dep_interfaces else "")
        )
        task = f"Write the file {spec.path} for this project: {objective}\nThis file's purpose: {spec.purpose}"
        try:
            result = self._longgen.generate(task, root / spec.path, key=f"{key}:{spec.path}", context=context)
        except RecoveryFailed as exc:
            return ModuleOutcome(spec.path, "failed", str(exc))
        if not result.valid:
            return ModuleOutcome(spec.path, "invalid", result.error, result.recoveries)
        state = analyze(result.source)
        defined = {f.name for f in state.functions} | {c.name for c in state.classes}
        wanted = {m.group(1) for i in spec.interfaces if (m := _NAME.search(i))}
        missing = sorted(wanted - defined)
        return ModuleOutcome(spec.path, "done", f"missing {missing}" if missing else "", result.recoveries, missing)

    def _integrate(self, root: Path, has_tests: bool) -> tuple[bool | None, str]:
        cmd = [sys.executable, "-m", "pytest", "-q"] if has_tests else [sys.executable, "-m", "compileall", "-q", "."]
        try:
            p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=self._timeout)
        except subprocess.TimeoutExpired:
            return False, "integration timed out"
        return p.returncode == 0, (p.stdout + p.stderr)[-3000:]

    @staticmethod
    def _save(path: Path, manifest: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()
