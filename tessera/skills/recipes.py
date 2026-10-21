"""Level 2: recipes — deterministic builders that produce a skill without the model.

A recipe module lives in `skills/recipes/` and exposes

    RECIPE = {"name": "find_files", "description": "...", "keywords": ["find", "search", "files"]}
    def build(request) -> str        # returns complete skill source

Recipes are trusted, shipped code, so the book imports them (unlike skills,
which are only parsed). New recipes are discovered by dropping a file in.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

from ..core.types import SkillRequest

log = logging.getLogger("tessera.skills.recipes")

DEFAULT_RECIPE_DIR = Path(__file__).with_name("recipes")


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    keywords: tuple[str, ...]
    build: Callable[[SkillRequest], str]


class RecipeBook:
    def __init__(self, folder: Path = DEFAULT_RECIPE_DIR) -> None:
        self._folder = folder
        self._recipes: dict[str, Recipe] = {}
        self.refresh()

    def refresh(self) -> None:
        self._recipes.clear()
        if not self._folder.exists():
            return
        for path in sorted(self._folder.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                mod = _load(path)
                meta = mod.RECIPE
                self._recipes[meta["name"]] = Recipe(
                    meta["name"], meta["description"], tuple(meta.get("keywords", ())), mod.build
                )
            except Exception as exc:  # a broken recipe must not disable the rest
                log.warning("recipe %s skipped: %s", path.name, exc)

    def get(self, name: str) -> Recipe | None:
        return self._recipes.get(name)

    def all(self) -> list[Recipe]:
        return list(self._recipes.values())


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"tessera_recipe_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
