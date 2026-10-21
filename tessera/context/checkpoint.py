"""Durable JSON checkpoints for interrupted work.

SRP: atomic save/load/list of checkpoint records. Shared by context
recovery (Phase 4) and long-running workflows (Phase 6). Temporary by
design: `discard()` after completion; nothing here enters long-term memory.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", key)[:120]
        return self.root / f"{safe}.json"

    def save(self, key: str, data: dict[str, Any]) -> Path:
        path = self._path(key)
        record = {**data, "_saved_at": time.time(), "_key": key}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)  # atomic on the same filesystem
        return path

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, prefix: str = "") -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if rec.get("_key", "").startswith(prefix):
                out.append(rec)
        return out

    def discard(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
