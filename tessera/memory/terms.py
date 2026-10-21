"""Deterministic term extraction for graph nodes — no model calls.

Natural language: content words (lower-cased, light plural stripping) plus
capitalised multi-word names ("Main Street", "GPT Luna").
Python source: identifiers from `ast` (imports, classes, functions).
"""

from __future__ import annotations

import ast
import re

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself just like me
    more most my myself no nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves yes ok okay please thanks thank hello hi hey really much
    many get got make made use used using one two three thing things something anything way lot let
    want need know think see say said tell go going went come came give take put still even back well
    sure maybe might must shall may every ever never always often sometimes today tomorrow yesterday""".split()
)

# Nouns whose plural is -ies but whose singular ends in -ie, not -y.
_IE_NOUNS = frozenset(
    "cookie movie pie tie lie die zombie rookie selfie hoodie freebie calorie genie prairie "
    "smoothie brownie goalie auntie sweetie techie newbie budgie collie eerie".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")
_PROPER = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+)\b")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")


def normalise(word: str) -> str:
    w = word.lower().strip("'-_")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-1] if w[:-1] in _IE_NOUNS else w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return w


def extract_terms(text: str, limit: int = 24) -> list[str]:
    terms: list[str] = []
    for m in _PROPER.finditer(text):
        terms.append(m.group(1).lower())
    for m in _WORD.finditer(text):
        w = normalise(m.group(0))
        if len(w) >= 3 and w not in STOPWORDS and not w.isdigit():
            terms.append(w)
    return list(dict.fromkeys(terms))[:limit]


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        while len(p) > max_chars:  # hard-wrap very long runs at a space
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def python_terms(source: str) -> list[str]:
    """Identifiers that matter for associations; empty if source doesn't parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name.lower())
        elif isinstance(node, ast.Import):
            names.extend(a.name.split(".")[0].lower() for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0].lower())
    return list(dict.fromkeys(names))
