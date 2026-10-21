"""Code-generation tools backed by context recovery (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from ..context.project_builder import PlanError, ProjectBuilder
from ..context.recovery import LongCodeGenerator, RecoveryFailed
from ..core.types import Observation
from ..execution.artifacts import WorkspaceManager
from ..skills.contract import SkillSpec
from .requirements import requirement_problems


def _within(base: Path, rel: str) -> Path:
    target = (base / rel).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError(f"path escapes the workspace: {rel}")
    return target


class WriteModuleTool:
    spec = SkillSpec(
        name="write_python_module",
        description="Write one (possibly long) Python file, surviving context-window limits; saved in the user's output folder",
        parameters={"path": "str - relative file name, e.g. parser.py", "description": "str - what the module must do"},
        origin="native",
    )

    def __init__(self, longgen: LongCodeGenerator, workspaces: WorkspaceManager, output_dir: Path | None = None) -> None:
        self._gen, self._ws, self._out = longgen, workspaces, output_dir

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        try:
            base = self._out or self._ws.for_task(task_id)   # user's output folder when configured
            base.mkdir(parents=True, exist_ok=True)
            path = _within(base, str(arguments.get("path", "module.py")))
            res = self._gen.generate(str(arguments.get("description") or purpose), path, key=f"{task_id}:{path.name}")
        except (ValueError, RecoveryFailed) as exc:
            return Observation("error", self.spec.name, stderr=str(exc))
        unmet = requirement_problems(res.source, purpose, path.suffix)
        ok = res.valid and not unmet
        return Observation(
            "success" if ok else "error", self.spec.name,
            result={"path": str(path), "lines": res.source.count("\n"), "syntax_valid": res.valid},
            stderr="\n".join(filter(None, [res.error] + unmet)), artifacts=(str(path),) if ok else (),
            notes=(f"saved to {path}", f"{len(res.rounds)} generation rounds, {res.recoveries} context recoveries"),
        )


class BuildProjectTool:
    spec = SkillSpec(
        name="build_project",
        description="Plan and write a multi-file Python project module by module, then run its tests",
        parameters={"description": "str - what to build", "folder": "str - relative folder name, default project"},
        origin="native",
    )

    def __init__(self, builder: ProjectBuilder, workspaces: WorkspaceManager, output_dir: Path | None = None) -> None:
        self._builder, self._ws, self._out = builder, workspaces, output_dir

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        try:
            base = self._out or self._ws.for_task(task_id)
            base.mkdir(parents=True, exist_ok=True)
            root = _within(base, str(arguments.get("folder", "project")))
            report = self._builder.build(str(arguments.get("description") or purpose), root, key=f"{task_id}:project")
        except (ValueError, PlanError, RecoveryFailed) as exc:
            return Observation("error", self.spec.name, stderr=str(exc))
        return Observation(
            "success" if report.ok else "error", self.spec.name,
            result={"root": str(root), "files": report.summary()},
            stdout=report.integration_output[-1500:],
            artifacts=tuple(str(root / m.path) for m in report.modules),
        )
