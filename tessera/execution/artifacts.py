"""Workspace management and artifact detection.

SRP: give each task an isolated working folder and report which files a run
created or modified there (by size + mtime snapshot, cheap and exact enough).
"""

from __future__ import annotations

import re
from pathlib import Path


class WorkspaceManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def for_task(self, task_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", task_id or "default")[:64]
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path


def snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for p in folder.rglob("*"):
        if p.is_file() and not p.name.startswith(".tessera_"):
            st = p.stat()
            out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


def changed_files(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    return sorted(p for p, sig in after.items() if before.get(p) != sig)
