"""Self-inspection and reversible self-modification.

SRP: propose → approve → back up → apply → validate (compile + focused
tests) → keep or roll back → report precisely.

Edits are SEARCH/REPLACE blocks, applied only if each SEARCH text occurs
exactly once — a small model's patch can't silently land in the wrong place.

Activation honesty: Tessera's core modules are already imported, so a patch
to them is *on disk, not active* until restart. Skills are loaded fresh in a
child process on every run, so a patched skill *is* active immediately. The
report says which applies.
"""

from __future__ import annotations

import difflib
import py_compile
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.interfaces import Approver, ModelProvider
from ..core.types import Message, Observation
from ..skills.contract import SkillSpec

PATCH_PROMPT = """Stage: selfpatch.
You improve one Python file. Make ONE focused improvement toward the goal.
Reply with one or more blocks exactly like:
<<<<<<< SEARCH
<exact existing lines>
=======
<replacement lines>
>>>>>>> REPLACE
Then one line: WHY: <what changed and why>. SEARCH text must match the file exactly."""

_BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", re.S)


class PatchError(ValueError):
    pass


@dataclass(frozen=True)
class Edit:
    search: str
    replace: str


def parse_edits(text: str) -> tuple[list[Edit], str]:
    edits = [Edit(s, r) for s, r in _BLOCK.findall(text)]
    why = re.search(r"^WHY:\s*(.+)$", text, re.M)
    if not edits:
        raise PatchError("no SEARCH/REPLACE blocks in reply")
    return edits, why.group(1).strip() if why else ""


def apply_edits(source: str, edits: list[Edit]) -> str:
    out = source
    for i, e in enumerate(edits, 1):
        n = out.count(e.search)
        if n != 1:
            raise PatchError(f"edit {i}: SEARCH text found {n} times (must be exactly once)")
        out = out.replace(e.search, e.replace, 1)
    return out


@dataclass
class PatchReport:
    file: str
    applied: bool
    rolled_back: bool
    backup: str
    diff: str
    why: str
    compiled: bool
    tests_passed: bool | None
    test_output: str
    active_now: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "applied" if self.applied and not self.rolled_back else "rolled back" if self.rolled_back else "not applied"
        tests = "n/a" if self.tests_passed is None else "passed" if self.tests_passed else "FAILED"
        active = "active now" if self.active_now else "on disk only — restart Tessera to activate"
        return f"{self.file}: {state}; compile={'ok' if self.compiled else 'FAILED'}; tests={tests}; {active}"


class SourceModifier:
    def __init__(self, project_root: Path, backup_root: Path, approver: Approver, *,
                 test_command: list[str] | None = None, test_timeout_s: int = 600,
                 hot_dirs: tuple[Path, ...] = ()) -> None:
        self.root = project_root.resolve()
        self.backups = backup_root
        self._approver = approver
        self._test_cmd = test_command or [sys.executable, "-m", "pytest", "-q", "-x"]
        self._timeout = test_timeout_s
        self._hot = tuple(p.resolve() for p in hot_dirs)  # files here are loaded fresh per run

    def _resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents:
            raise PatchError(f"{rel} is outside the project")
        if not p.is_file() or p.suffix != ".py":
            raise PatchError(f"{rel} is not an existing Python file")
        return p

    def apply(self, rel_path: str, edits: list[Edit], why: str = "", tests: list[str] | None = None) -> PatchReport:
        path = self._resolve(rel_path)
        before = path.read_text(encoding="utf-8")
        after = apply_edits(before, edits)
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            f"a/{rel_path}", f"b/{rel_path}"))
        rel = str(path.relative_to(self.root))
        if not self._approver.approve("apply_patch", f"{rel}: {why or 'patch'}\n{diff[:1500]}"):
            return PatchReport(rel, False, False, "", diff, why, False, None, "", False, ["patch not approved"])

        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 10_000:04d}"
        backup = self.backups / stamp / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        path.write_text(after, encoding="utf-8")

        compiled, test_ok, out = True, None, ""
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compiled, out = False, str(exc)
        if compiled:
            cmd = self._test_cmd + (tests or [])
            try:
                p = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=self._timeout)
                test_ok, out = p.returncode == 0, (p.stdout + p.stderr)[-3000:]
            except subprocess.TimeoutExpired:
                test_ok, out = False, "tests timed out"

        rolled_back = not compiled or test_ok is False
        if rolled_back:
            shutil.copy2(backup, path)
        active = not rolled_back and any(d in path.parents for d in self._hot)
        return PatchReport(rel, True, rolled_back, str(backup), diff, why, compiled, test_ok, out, active)

    def rollback(self, backup_path: str) -> str:
        b = Path(backup_path).resolve()
        if self.backups.resolve() not in b.parents:
            raise PatchError("not a Tessera backup")
        rel = b.relative_to(self.backups.resolve()).parts[1:]
        target = self.root.joinpath(*rel)
        shutil.copy2(b, target)
        return str(target)


class ImproveSourceTool:
    spec = SkillSpec(
        name="improve_source",
        description="Read one of Tessera's own source files, make one focused, tested, reversible improvement",
        parameters={"path": "str - file relative to the project root", "goal": "str - what to improve",
                    "tests": "list[str] - optional pytest targets, e.g. ['tests/test_router.py']"},
        origin="native",
    )

    def __init__(self, model: ModelProvider, modifier: SourceModifier, *, max_tokens: int = 900,
                 max_file_chars: int = 9000) -> None:
        self._model, self._mod = model, modifier
        self._max_tokens, self._max_chars = max_tokens, max_file_chars

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        rel = str(arguments.get("path", ""))
        try:
            path = self._mod._resolve(rel)
        except PatchError as exc:
            return Observation("error", self.spec.name, stderr=str(exc))
        source = path.read_text(encoding="utf-8")
        if len(source) > self._max_chars:
            return Observation("error", self.spec.name,
                               stderr=f"{rel} is {len(source)} chars; too large for one patch prompt — narrow the goal to a function")
        reply = self._model.generate(
            [Message.system(PATCH_PROMPT), Message.user(f"GOAL: {arguments.get('goal') or purpose}\n\nFILE {rel}:\n{source}")],
            max_output_tokens=self._max_tokens, temperature=0.2,
        ).text
        try:
            edits, why = parse_edits(reply)
            report = self._mod.apply(rel, edits, why, list(arguments.get("tests") or []))
        except PatchError as exc:
            return Observation("error", self.spec.name, stderr=f"patch rejected: {exc}", notes=("no files were changed",))
        ok = report.applied and not report.rolled_back
        return Observation(
            "success" if ok else ("denied" if not report.applied else "error"), self.spec.name,
            result=report.summary(), stdout=report.diff[-2500:], stderr="" if ok else report.test_output[-1500:],
            artifacts=(report.backup,) if report.backup else (),
            notes=tuple(report.notes) + ((f"why: {report.why}",) if report.why else ()),
        )
