"""Runs one skill file in a child Python process and observes it.

SRP: process mechanics — spawn, timeout, capture stdout/stderr, read the
harness result, detect artifacts. It does not decide what to do about a
failure; recovery.py does.

Success requires *both* exit code 0 and an `ok: true` result file written by
the harness. A process that exits 0 without a result file is an error.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import changed_files, snapshot

HARNESS = Path(__file__).with_name("harness.py").resolve()
OUTPUT_LIMIT = 20_000


@dataclass(frozen=True)
class RunResult:
    ok: bool
    result: Any
    stdout: str
    stderr: str
    exit_code: int | None
    duration_ms: int
    artifacts: tuple[str, ...]
    error_type: str = ""
    message: str = ""
    traceback: str = ""
    missing_module: str = ""
    timed_out: bool = False


class ProcessRunner:
    def __init__(self, python: str = sys.executable, timeout_s: float = 60.0, extra_env: dict | None = None) -> None:
        self._python = python
        self._timeout = timeout_s
        self._extra_env = extra_env or {}

    @property
    def timeout_s(self) -> float:
        return self._timeout

    def run(self, skill_path: Path, arguments: dict, cwd: Path, timeout_s: float | None = None) -> RunResult:
        cwd = Path(cwd).resolve()  # the child runs *inside* cwd, so every path we pass must be absolute
        cwd.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        args_file = cwd / f".tessera_args_{tag}.json"
        result_file = cwd / f".tessera_result_{tag}.json"
        args_file.write_text(json.dumps(arguments), encoding="utf-8")
        before = snapshot(cwd)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", **self._extra_env}
        cmd = [self._python, "-I", str(HARNESS), str(skill_path.resolve()), str(args_file), str(result_file)]

        start = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, capture_output=True, text=True,
                timeout=timeout_s or self._timeout, encoding="utf-8", errors="replace",
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code = None
        duration = int((time.perf_counter() - start) * 1000)

        payload: dict = {}
        try:
            if result_file.exists():
                payload = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"ok": False, "error_type": "HarnessError", "message": "unreadable result file"}
        finally:
            for f in (args_file, result_file):
                f.unlink(missing_ok=True)

        artifacts = tuple(changed_files(before, snapshot(cwd)))
        ok = bool(payload.get("ok")) and code == 0 and not timed_out
        if timed_out:
            payload = {"error_type": "TimeoutError", "message": f"skill exceeded {timeout_s or self._timeout}s"}
        elif not payload:
            payload = {"error_type": "HarnessError", "message": f"process exited {code} without a result"}

        return RunResult(
            ok=ok,
            result=payload.get("result"),
            stdout=stdout[-OUTPUT_LIMIT:],
            stderr=stderr[-OUTPUT_LIMIT:],
            exit_code=code,
            duration_ms=duration,
            artifacts=artifacts,
            error_type=payload.get("error_type", ""),
            message=payload.get("message", ""),
            traceback=payload.get("traceback", ""),
            missing_module=payload.get("missing_module", ""),
            timed_out=timed_out,
        )
