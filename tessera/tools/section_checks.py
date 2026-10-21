"""Per-language knowledge for section-by-section file building (non-HTML files).

SRP: for a file type, answer three deterministic questions — no model calls:
  * which names does this text define?        → the contract later sections receive
  * is this one section valid on its own?      → checked right after it's written
  * do sections clash with each other?         → duplicate definitions

Sections are complete units (whole functions, classes, CSS rules, paragraphs),
so most languages can be checked one section at a time. Add a language by adding
an entry to LANGUAGES; nothing else changes.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable

from .validators import bracket_problems, js_problems

_JS_DEFS = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s*\*?\s*([A-Za-z_$][\w$]*)|"
                      r"class\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*))", re.M)
_MD_HEADINGS = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


def _py_defs(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {n.name}({ast.unparse(n.args)})")
        elif isinstance(n, ast.ClassDef):
            meths = [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            out.append(f"class {n.name}" + (f" (methods: {', '.join(meths)})" if meths else ""))
        elif isinstance(n, ast.Assign):
            out += [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.append(ast.unparse(n))
    return out


def _py_names(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _py_check(text: str) -> list[str]:
    try:
        ast.parse(text)
        return []
    except SyntaxError as exc:
        return [f"Python: {exc.msg} at line {exc.lineno} of this section"]


def _js_names(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in _JS_DEFS.finditer(text)]


def _css_selectors(text: str) -> list[str]:
    from .html_builder import css_rules

    return [sel for sel, _ in css_rules(text) if sel]


def _md_headings(text: str) -> list[str]:
    return [f"{h} {t}" for h, t in _MD_HEADINGS.findall(text)]


@dataclass(frozen=True)
class Language:
    label: str
    defines: Callable[[str], list[str]]          # contract text for later sections
    names: Callable[[str], list[str]]            # identifiers that must not be redefined
    check: Callable[[str], list[str]]            # problems in one section on its own
    unit: str                                    # what a "complete unit" is, for the prompt


LANGUAGES: dict[str, Language] = {
    ".py": Language("Python", _py_defs, _py_names, _py_check, "whole functions/classes"),
    ".js": Language("JavaScript", _js_names, _js_names, lambda t: js_problems(t), "whole functions/statements"),
    ".mjs": Language("JavaScript", _js_names, _js_names, lambda t: js_problems(t), "whole functions/statements"),
    ".css": Language("CSS", _css_selectors, lambda t: [], lambda t: bracket_problems(t, "CSS"), "whole CSS rules"),
    ".md": Language("Markdown", _md_headings, lambda t: [], lambda t: [], "whole paragraphs/sections"),
}
GENERIC = Language("text", lambda t: [], lambda t: [], lambda t: [], "complete paragraphs or blocks")


def language_for(suffix: str) -> Language:
    return LANGUAGES.get(suffix.lower(), GENERIC)


def redefinitions(texts: list[str], lang: Language) -> dict[int, list[str]]:
    """Section index → problems for names an earlier section already defined."""
    seen: dict[str, int] = {}
    out: dict[int, list[str]] = {}
    for i, t in enumerate(texts):
        for n in lang.names(t):
            if n in seen and seen[n] != i:
                out.setdefault(i, []).append(
                    f"'{n}' is already defined in section {seen[n] + 1}; reuse it instead of defining it again")
            else:
                seen.setdefault(n, i)
    return out
