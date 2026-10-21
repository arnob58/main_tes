"""Discovers skills from folders and from in-process registrations.

SRP: know which skills exist and their metadata. No resolution heuristics
(resolver.py) and no execution (runtime.py).

OCP: dropping `weather.py` into any scanned folder makes it available on the
next `refresh()` — no import statements or registry edits. Native tools
(Python objects that need live services, e.g. the desktop loop) are added
with `register_native()` in the composition root, never in core code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.types import Observation
from .contract import ContractError, SkillSpec, parse_skill_source

log = logging.getLogger("tessera.skills")


class NativeTool(Protocol):
    spec: SkillSpec

    def invoke(self, arguments: dict[str, Any], *, task_id: str, purpose: str) -> Observation: ...


@dataclass(frozen=True)
class SkillSource:
    folder: Path
    origin: str


class SkillRegistry:
    def __init__(self, sources: list[SkillSource]) -> None:
        self._sources = sources
        self._file_skills: dict[str, SkillSpec] = {}
        self._native: dict[str, NativeTool] = {}
        self._mtimes: dict[Path, int] = {}
        self.invalid: dict[str, str] = {}  # path → reason, for diagnostics
        self.refresh()

    def refresh(self) -> list[str]:
        """Rescan folders; return names that are new since the last scan."""
        before = set(self._file_skills)
        found: dict[str, SkillSpec] = {}
        self.invalid.clear()
        for src in self._sources:
            if not src.folder.exists():
                continue
            for path in sorted(src.folder.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                try:
                    spec = parse_skill_source(
                        path.read_text(encoding="utf-8"), default_name=path.stem, path=path, origin=src.origin
                    )
                except (ContractError, OSError, UnicodeDecodeError) as exc:
                    self.invalid[str(path)] = str(exc)
                    continue
                if spec.name in found:  # earlier sources (builtin) win over later ones
                    log.warning("duplicate skill %s at %s ignored", spec.name, path)
                    continue
                found[spec.name] = spec
        self._file_skills = found
        return sorted(set(found) - before)

    def register_native(self, tool: NativeTool) -> None:
        self._native[tool.spec.name] = tool

    def get(self, name: str) -> SkillSpec | None:
        if name in self._native:
            return self._native[name].spec
        return self._file_skills.get(name)

    def native(self, name: str) -> NativeTool | None:
        return self._native.get(name)

    def all(self) -> list[SkillSpec]:
        return [t.spec for t in self._native.values()] + list(self._file_skills.values())

    def __len__(self) -> int:
        return len(self._file_skills) + len(self._native)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None
