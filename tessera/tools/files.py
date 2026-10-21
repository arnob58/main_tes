"""write_file: save any text file the user asked for, somewhere they can find it.

In-process native tool (not a child-process skill) because overwriting or
writing outside the output folder must go through the Approver.

Relative paths go under `output_directory` (default ~/Documents/Tessera).
Absolute or ~ paths outside that folder need approval; so does overwriting
an existing file. Nothing is deleted.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.interfaces import Approver
from ..core.types import Observation
from ..skills.contract import SkillSpec
from .requirements import requirement_problems
from .validators import validate_file

MAX_BYTES = 2_000_000


class WriteFileTool:
    spec = SkillSpec(
        name="write_file",
        description=("Save a text file (HTML, CSS, JS, Markdown, JSON, Python, …). Put the file content in a "
                     "fenced code block directly after the ACTION line. Relative paths are saved in the "
                     "user's Tessera output folder"),
        parameters={"path": "str - e.g. site_tracker/tracker.html", "content": "str - from the code block"},
        origin="native",
    )

    def __init__(self, output_dir: str | Path, approver: Approver) -> None:
        self.output_dir = Path(os.path.expanduser(str(output_dir))).resolve()
        self._approver = approver

    def _target(self, raw: str) -> Path:
        p = Path(os.path.expanduser(raw.strip()))
        return (p if p.is_absolute() else self.output_dir / p).resolve()

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        raw_path = str(arguments.get("path") or "").strip()
        content = arguments.get("content")
        if not raw_path:
            return Observation("error", self.spec.name, stderr="missing 'path'")
        if not isinstance(content, str) or not content.strip():
            return Observation("error", self.spec.name,
                               stderr="missing content: put the file in a fenced code block right after the ACTION line")
        data = content if content.endswith("\n") else content + "\n"
        if len(data.encode("utf-8")) > MAX_BYTES:
            return Observation("error", self.spec.name, stderr=f"content exceeds {MAX_BYTES} bytes")

        target = self._target(raw_path)
        problems = validate_file(target.name, data) + requirement_problems(data, purpose, target.suffix)
        if problems:
            return Observation(
                "error", self.spec.name,
                stderr="NOT SAVED — the file has bugs:\n- " + "\n- ".join(problems[:8]),
                notes=("fix every problem listed, then call write_file again with the COMPLETE corrected file",),
            )
        inside = self.output_dir == target or self.output_dir in target.parents
        if not inside and not self._approver.approve("write_outside_output", f"write {target}"):
            return Observation("denied", self.spec.name, result=f"writing outside {self.output_dir} was not approved")
        existed = target.exists()
        if existed:
            if target.is_dir():
                return Observation("error", self.spec.name, stderr=f"{target} is a folder")
            if target.read_text(encoding="utf-8", errors="replace") == data:
                return Observation("success", self.spec.name, result={"path": str(target), "unchanged": True})
            if not self._approver.approve("overwrite_file", f"overwrite existing {target}"):
                return Observation("denied", self.spec.name, result=f"{target} already exists; overwrite not approved")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data, encoding="utf-8")
        return Observation(
            "success", self.spec.name,
            result={"path": str(target), "bytes": len(data.encode("utf-8")), "lines": data.count("\n"),
                    "overwrote": existed},
            artifacts=(str(target),),
            notes=(f"saved to {target}",),
        )
