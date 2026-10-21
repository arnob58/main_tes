"""Python-specific fixes and checks for generated files (no model calls).

From real use (web_server.py): section 1 wrote the whole program including the
start-up call `run_server()`; once duplicates were removed, that call sat ABOVE the
functions it needs → NameError on launch. And the code caught a failed import and
silently "mocked" Tessera instead of failing.

  reorder_entrypoints   move `if __name__ == "__main__":` blocks and bare top-level
                        calls to this file's own functions to the end of the file
  mock_problems         flag `except ImportError:` handlers that fake the module
  import_problems       actually import the finished file in a child process (with
                        __name__ != "__main__", so start-up code doesn't run) to catch
                        crash-at-start bugs: NameError, bad imports, typos
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

IMPORT_TIMEOUT_S = 20


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    parts = [node.test.left] + list(node.test.comparators)
    return any(isinstance(p, ast.Name) and p.id == "__name__" for p in parts) and \
        any(isinstance(p, ast.Constant) and p.value == "__main__" for p in parts)


def _defined(tree: ast.Module) -> set[str]:
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def reorder_entrypoints(texts: list[str]) -> tuple[list[str], list[str]]:
    """Move entry-point statements from earlier sections to the end of the last one."""
    if len(texts) < 2:
        return texts, []
    trees = []
    for t in texts:
        try:
            trees.append(ast.parse(t))
        except SyntaxError:
            trees.append(None)
    file_funcs = set().union(*(_defined(t) for t in trees if t))
    out, moved, notes = list(texts), [], []
    for i, tree in enumerate(trees[:-1]):
        if tree is None:
            continue
        lines = out[i].splitlines(keepends=True)
        spans = []
        for node in tree.body:
            call_to_own = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                           and isinstance(node.value.func, ast.Name) and node.value.func.id in file_funcs)
            if _is_main_guard(node) or call_to_own:
                spans.append((node.lineno, node.end_lineno))
        for start, end in sorted(spans, reverse=True):
            moved.insert(0, "".join(lines[start - 1 : end]))
            del lines[start - 1 : end]
        if spans:
            out[i] = "".join(lines)
            notes.append(f"moved {len(spans)} start-up statement(s) from section {i + 1} to the end of the file")
    if moved:
        last = out[-1].rstrip("\n")
        out[-1] = last + "\n\n\n" + "\n".join(m.rstrip("\n") for m in moved) + "\n"
    return out, notes


def mock_problems(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            names = []
            if isinstance(h.type, ast.Name):
                names = [h.type.id]
            elif isinstance(h.type, ast.Tuple):
                names = [e.id for e in h.type.elts if isinstance(e, ast.Name)]
            if not {"ImportError", "ModuleNotFoundError"} & set(names):
                continue
            fakes = [s for s in h.body if isinstance(s, (ast.ClassDef, ast.FunctionDef, ast.Assign, ast.ImportFrom))]
            if fakes:
                problems.append(f"Python: line {h.lineno} catches a failed import and substitutes a fake — don't mock "
                                "missing modules; let the import fail (or print a clear error and exit)")
    return problems


_CHECK = r"""
import runpy, sys, traceback
path = sys.argv[1]
for extra in sys.argv[2:]:
    sys.path.insert(0, extra)
try:
    runpy.run_path(path, run_name="__tessera_import_check__")
except SystemExit:
    pass
except BaseException as exc:
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == path:
            line = frame.lineno
    print(f"TESSERA_IMPORT_ERROR|{type(exc).__name__}|{line}|{exc}")
    sys.exit(1)
"""


def import_problems(path: Path, extra_paths: tuple[Path, ...] = (), timeout_s: int = IMPORT_TIMEOUT_S
                    ) -> tuple[list[str], list[str]]:
    """Import the file in a child process. Returns (blocking problems, warnings)."""
    path = Path(path).resolve()
    check = path.with_name(f".tessera_check_{path.stem}.py")
    check.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    args = [sys.executable, "-c", _CHECK, str(check), str(path.parent)] + [str(p) for p in extra_paths]
    try:
        p = subprocess.run(args, cwd=path.parent, capture_output=True, text=True, timeout=timeout_s,
                           stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return [f"Import check: the file was still running after {timeout_s}s just from being imported — put start-up "
                "code (servers, loops, input()) inside if __name__ == \"__main__\":"], []
    finally:
        check.unlink(missing_ok=True)
    m = re.search(r"TESSERA_IMPORT_ERROR\|(\w+)\|(\w+)\|(.*)", p.stdout)
    if not m:
        return [], []
    kind, line, msg = m.group(1), m.group(2), m.group(3).strip()
    where = f" (line {line})" if line != "None" else ""
    if kind == "ModuleNotFoundError":
        return [], [f"needs a package that isn't installed here: {msg}{where}"]
    return [f"Import check: {kind}: {msg}{where}"], []
