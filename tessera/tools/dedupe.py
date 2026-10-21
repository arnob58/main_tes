"""Deterministic removal of duplicate top-level definitions across file sections.

Real failure (twice): Gemma wrote the whole program into section 1, then section 2
defined the same function again. Asking the model to "not redefine it" targeted the
wrong section and never converged. Instead, Tessera now decides which copy stays and
deletes the other with a parser — no model call.

Which copy stays (the "owner"):
  1. the section whose planned name matches the identifier (section 'client_handler'
     owns def client_handler) — that is where the outline put it;
  2. otherwise the LAST copy (later sections are written with more context).
"""

from __future__ import annotations

import ast
import re

_JS_FUNC = r"(?m)^[ \t]*(?:export\s+)?(?:async\s+)?function\s*\*?\s*{name}\s*\("
_JS_CLASS = r"(?m)^[ \t]*(?:export\s+)?class\s+{name}\b"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _py_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    """name → [(start_line, end_line)] (1-based, inclusive), decorators included."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out: dict[str, list[tuple[int, int]]] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in n.decorator_list] + [n.lineno])
            out.setdefault(n.name, []).append((start, n.end_lineno))
    return out


def _remove_py(text: str, name: str) -> str:
    spans = _py_spans(text).get(name)
    if not spans:
        return text
    lines = text.splitlines(keepends=True)
    for start, end in sorted(spans, reverse=True):
        del lines[start - 1 : end]
    return re.sub(r"\n{3,}", "\n\n", "".join(lines))


def _js_block_end(text: str, open_brace: int) -> int:
    depth, i, n = 0, open_brace, len(text)
    while i < n:
        c = text[i]
        if c in "\"'`":
            q, i = c, i + 1
            while i < n and text[i] != q:
                i += 2 if text[i] == "\\" else 1
        elif text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                return n
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _js_names(text: str) -> list[str]:
    return re.findall(r"(?m)^[ \t]*(?:export\s+)?(?:async\s+)?(?:function\s*\*?\s*|class\s+)([A-Za-z_$][\w$]*)", text)


def _remove_js(text: str, name: str) -> str:
    for pat in (_JS_FUNC, _JS_CLASS):
        m = re.search(pat.format(name=re.escape(name)), text)
        if m:
            brace = text.find("{", m.end() - 1)
            if brace < 0:
                return text
            end = _js_block_end(text, brace)
            line_start = text.rfind("\n", 0, m.start()) + 1
            return text[:line_start] + text[end:].lstrip("\n")
    return text


def defined_names(text: str, suffix: str) -> list[str]:
    if suffix == ".py":
        return list(_py_spans(text))
    if suffix in (".js", ".mjs"):
        return _js_names(text)
    return []


def dedupe_sections(names: list[str], texts: list[str], suffix: str) -> tuple[list[str], list[str]]:
    """Return (new_texts, notes). `names` are the planned section names."""
    suffix = suffix.lower()
    if suffix not in (".py", ".js", ".mjs"):
        return texts, []
    remover = _remove_py if suffix == ".py" else _remove_js
    where: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        for n in dict.fromkeys(defined_names(t, suffix)):
            where.setdefault(n, []).append(i)
    texts = list(texts)
    notes: list[str] = []
    for name, idxs in where.items():
        if len(idxs) < 2:
            continue
        owner = next((i for i in idxs if _norm(name) and _norm(name) in _norm(names[i])), idxs[-1])
        for i in idxs:
            if i != owner:
                texts[i] = remover(texts[i], name)
        removed = [names[i] for i in idxs if i != owner]
        notes.append(f"kept '{name}' from section '{names[owner]}', removed the copy in {', '.join(removed)}")
    return texts, notes
