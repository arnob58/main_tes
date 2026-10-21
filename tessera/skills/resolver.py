"""Maps a requested capability to an existing skill (L1), a recipe (L2) or generation (L3).

SRP: matching only. Deterministic token overlap over names, descriptions and
keywords — cheap, explainable, and no model call. A fuzzy match is accepted
only if the request's arguments fit the skill's declared parameters, so
"read_pdf(path)" never silently resolves to "read_url(url)".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.types import SkillRequest
from .contract import SkillSpec, normalise_name
from .recipes import Recipe, RecipeBook
from .registry import SkillRegistry

_TOK = re.compile(r"[a-z0-9]+")
_SYNONYMS = {
    "search": "find", "locate": "find", "lookup": "find", "discover": "find",
    "open": "read", "load": "read", "extract": "read", "get": "read", "fetch": "read",
    "file": "files", "document": "doc", "documents": "doc", "dir": "directory", "folder": "directory",
    "list": "list", "ls": "list", "code": "source", "sources": "source",
}
_IGNORE = {"a", "an", "the", "of", "to", "for", "and", "from", "in", "on", "my", "skill", "tool", "text", "content", "contents"}


def tokens(text: str) -> set[str]:
    out = set()
    for t in _TOK.findall(text.lower().replace("_", " ")):
        if t in _IGNORE:
            continue
        out.add(_SYNONYMS.get(t, t))
    return out


@dataclass(frozen=True)
class Resolution:
    level: int                     # 1 skill, 2 recipe, 3 generate
    skill: SkillSpec | None = None
    recipe: Recipe | None = None
    reason: str = ""


class SkillResolver:
    def __init__(self, registry: SkillRegistry, recipes: RecipeBook | None, *, fuzzy_threshold: float = 0.6) -> None:
        self._registry = registry
        self._recipes = recipes
        self._threshold = fuzzy_threshold

    def resolve(self, request: SkillRequest) -> Resolution:
        name = normalise_name(request.skill)
        exact = self._registry.get(name)
        if exact is not None:
            return Resolution(1, skill=exact, reason="exact name")

        want = tokens(name)
        best, best_score = None, 0.0
        for spec in self._registry.all():
            score = _overlap(want, tokens(spec.name)) * 0.7 + _overlap(want, tokens(spec.description)) * 0.3
            if score > best_score and _args_fit(request.arguments, spec):
                best, best_score = spec, score
        if best is not None and best_score >= self._threshold:
            return Resolution(1, skill=best, reason=f"similar to '{best.name}' ({best_score:.2f})")

        if self._recipes is not None:
            if (r := self._recipes.get(name)) is not None:
                return Resolution(2, recipe=r, reason="recipe name")
            rbest, rscore = None, 0.0
            for r in self._recipes.all():
                score = _overlap(want, tokens(r.name) | {_SYNONYMS.get(k, k) for k in r.keywords})
                if score > rscore:
                    rbest, rscore = r, score
            if rbest is not None and rscore >= self._threshold:
                return Resolution(2, recipe=rbest, reason=f"recipe '{rbest.name}' ({rscore:.2f})")

        return Resolution(3, reason="no matching skill or recipe")


def _overlap(want: set[str], have: set[str]) -> float:
    return len(want & have) / len(want) if want else 0.0


def _args_fit(args: dict, spec: SkillSpec) -> bool:
    if not spec.parameters:
        return True
    return set(args) <= set(spec.parameters)
