"""The skill contract, checked statically.

A skill module exposes either
    DESCRIPTION = "…"           and   def run(**kwargs): …
or
    SKILL = {"name": …, "description": …, "parameters": {…}, "dependencies": […]}
                                and   def run(**kwargs): …

Metadata is read with `ast.literal_eval` — the module is *never imported* to
discover it. That keeps discovery side-effect free and means a malicious or
broken generated file can't run code just by being in the skills folder.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    path: Path | None = None
    origin: str = "generated"   # builtin | generated | workspace | native
    imports: tuple[str, ...] = ()

    def signature(self) -> str:
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({params}) — {self.description}"


def normalise_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if n and n[0].isdigit():
        n = "skill_" + n
    return n[:64]


def parse_skill_source(source: str, *, default_name: str, path: Path | None = None, origin: str = "generated") -> SkillSpec:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ContractError(f"syntax error at line {exc.lineno}: {exc.msg}") from exc

    meta: dict = {}
    description = None
    has_run = False
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            if isinstance(node, ast.AsyncFunctionDef):
                raise ContractError("run() must be a regular function, not async")
            has_run = True
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target in ("SKILL", "DESCRIPTION"):
                try:
                    value = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise ContractError(f"{target} must be a literal") from exc
                if target == "SKILL":
                    if not isinstance(value, dict):
                        raise ContractError("SKILL must be a dict")
                    meta = value
                else:
                    description = value
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.append(node.module.split(".")[0])

    if not has_run:
        raise ContractError("module must define a top-level run(**kwargs)")
    description = meta.get("description", description)
    if not isinstance(description, str) or not description.strip():
        raise ContractError("a non-empty DESCRIPTION or SKILL['description'] is required")
    name = normalise_name(str(meta.get("name", default_name)))
    if not NAME_RE.match(name):
        raise ContractError(f"invalid skill name {name!r}")
    params = meta.get("parameters", {}) or {}
    if not isinstance(params, dict):
        raise ContractError("SKILL['parameters'] must be a dict of name → type description")
    deps = meta.get("dependencies", ()) or ()
    return SkillSpec(
        name=name,
        description=description.strip(),
        parameters={str(k): str(v) for k, v in params.items()},
        dependencies=tuple(str(d) for d in deps),
        path=path,
        origin=origin,
        imports=tuple(dict.fromkeys(imports)),
    )
