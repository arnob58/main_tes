"""Static risk scan of skill source before it runs.

SRP: list operations in a module that deserve the user's approval —
deleting or moving files, spawning shells, writing outside the workspace is
hard to prove statically, so destructive calls are what we look for. This is
a speed bump for honest mistakes by a small model, not a sandbox against an
adversary; the README says so.
"""

from __future__ import annotations

import ast

RISKY_CALLS = {
    ("os", "remove"), ("os", "unlink"), ("os", "rmdir"), ("os", "removedirs"), ("os", "system"),
    ("os", "rename"), ("os", "replace"), ("os", "kill"), ("os", "popen"),
    ("shutil", "rmtree"), ("shutil", "move"),
    ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "Popen"), ("subprocess", "check_call"),
    ("subprocess", "check_output"),
}
RISKY_METHODS = {"unlink", "rmdir", "rmtree"}
RISKY_IMPORTS = {"ctypes", "winreg"}


def risky_operations(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and (f.value.id, f.attr) in RISKY_CALLS:
                    found.append(f"{f.value.id}.{f.attr}()")
                elif f.attr in RISKY_METHODS:
                    found.append(f".{f.attr}()")
            elif isinstance(f, ast.Name) and f.id in {"eval", "exec", "__import__"}:
                found.append(f"{f.id}()")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            found.extend(f"import {n}" for n in names if n.split(".")[0] in RISKY_IMPORTS)
    return sorted(set(found))
