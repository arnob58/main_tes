"""Files the user names are read by Tessera itself, before the model starts.

Real failure: asked to "read app.py and __main__.py", Gemma skipped the tool call and
answered from memory — describing a different program. Reading is deterministic, so
Python does it: every existing text-file path in the message is read and handed to the
agents as an observation. A path that looks like a file but doesn't exist becomes an
error observation, so the reply must say it couldn't be read rather than guess.

Reads are also remembered for ONE following turn, so "using what you learned from
app.py…" works in the next message.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .types import Observation

TEXT_EXT = {".py", ".js", ".mjs", ".ts", ".html", ".htm", ".css", ".json", ".md", ".txt", ".toml", ".yaml", ".yml",
            ".ini", ".cfg", ".csv", ".xml", ".sql", ".sh", ".ps1", ".bat", ".java", ".c", ".cpp", ".h", ".cs", ".go",
            ".rs", ".rb", ".php", ".log"}
_PATH = re.compile(
    r"""(?:"([^"\n]+\.[A-Za-z0-9]{1,5})"|'([^'\n]+\.[A-Za-z0-9]{1,5})'|"""   # quoted
    r"""((?:[A-Za-z]:[\\/]|~[\\/]|\.{0,2}[\\/])?(?:[\w.\-]+[\\/])*[\w.\-]+\.[A-Za-z0-9]{1,5}))(?=[\s,;:)!?]|\.(?:\s|$)|$)""")


_READ_VERBS = re.compile(r"\b(read|open|look at|explain|analy[sz]e|review|summari[sz]e|inspect|check|learn|using|based on)\b", re.I)


@dataclass
class FileReferenceReader:
    max_chars: int = 12000
    max_files: int = 4
    base_dirs: tuple[Path, ...] = field(default_factory=lambda: (Path.cwd(),))
    _previous: list[Observation] = field(default_factory=list)

    def candidates(self, text: str) -> list[str]:
        out = []
        for m in _PATH.finditer(text):
            raw = next(g for g in m.groups() if g).strip().rstrip(".")
            if Path(raw).suffix.lower() in TEXT_EXT and raw not in out:
                out.append(raw)
        return out

    def _resolve(self, raw: str) -> Path | None:
        p = Path(os.path.expanduser(raw))
        if p.is_absolute():
            return p if p.is_file() else None
        for base in self.base_dirs:
            if (base / p).is_file():
                return (base / p).resolve()
        return None

    def read_mentioned(self, text: str) -> list[Observation]:
        obs: list[Observation] = []
        wants_reading = bool(_READ_VERBS.search(text))
        for raw in self.candidates(text)[: self.max_files]:
            path = self._resolve(raw)
            looks_absolute = bool(re.match(r"^([A-Za-z]:[\\/]|~[\\/]|/)", raw))
            if path is None:
                # A missing path in "build site/x.html" is a file to create, not a failed read.
                if wants_reading and looks_absolute:
                    obs.append(Observation("error", f"file:{raw}", stderr=f"File not found: {raw}. It was NOT read.",
                                           notes=("tell the user this file could not be read; do not guess its content",)))
                continue
            try:
                data = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                obs.append(Observation("error", f"file:{raw}", stderr=f"Could not read {path}: {exc}"))
                continue
            clipped = len(data) > self.max_chars
            obs.append(Observation(
                "success", f"file:{path}",
                result=data[: self.max_chars] + ("\n…[file continues]" if clipped else ""),
                notes=(f"read by Tessera before answering ({len(data)} chars{', truncated' if clipped else ''})",),
            ))
        return obs

    # ------------------------------------------------------------- carry-over
    def previous_reads(self) -> list[Observation]:
        return [Observation(o.status, o.source, result=o.result, stderr=o.stderr,
                            notes=("read in your previous turn",)) for o in self._previous]

    def remember(self, observations: list[Observation]) -> None:
        """Keep this turn's successful file reads (by Tessera or by reading skills) for the next turn."""
        keep = [o for o in observations if o.status == "success" and
                (o.source.startswith("file:") or o.source.startswith("read_"))]
        self._previous = keep[: self.max_files]
