"""Deterministic JavaScript repair + error explanations (patch 11).

Real failure: Gemma named variables after CSS classes — `const chat-history = …` —
and 12 fix rounds failed because she only ever saw "Missing initializer in const
declaration (line 1)". A hyphen can never be part of a JS name, so Tessera renames
those itself (chat-history → chatHistory), never touching strings, templates or
comments (so '.chat-history' selectors stay intact).

Also: `pinpoint()` shows the failing line with a pointer, and `explain()` turns
common cryptic errors into a plain hint, so the model fixes the real cause.
"""

from __future__ import annotations

import re

_HYPHEN_DECL = re.compile(
    r"\b(?:const|let|var|function\*?|class)\s+([A-Za-z_$][\w$]*(?:-[A-Za-z_$][\w$]*)+)")
_HYPHEN_PARAM = re.compile(r"\(([^()]*)\)\s*(?:=>|\{)")


def _camel(name: str) -> str:
    head, *rest = name.split("-")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def _code_spans(code: str) -> list[tuple[int, int]]:
    """(start, end) ranges that are real code — outside strings, templates and comments."""
    spans, i, start, n = [], 0, 0, len(code)
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c in "'\"`" or (c == "/" and nxt in "/*"):
            spans.append((start, i))
            if c == "/" and nxt == "/":
                end = code.find("\n", i)
                i = n if end < 0 else end
            elif c == "/":
                end = code.find("*/", i + 2)
                i = n if end < 0 else end + 2
            else:
                i += 1
                while i < n and code[i] != c:
                    if code[i] == "\\":
                        i += 1
                    elif c != "`" and code[i] == "\n":
                        break
                    i += 1
                i += 1
            start = i
            continue
        i += 1
    spans.append((start, n))
    return [(a, b) for a, b in spans if b > a]


def _declared_hyphen_names(code: str) -> set[str]:
    names = set()
    for a, b in _code_spans(code):
        chunk = code[a:b]
        names.update(m.group(1) for m in _HYPHEN_DECL.finditer(chunk))
        for m in _HYPHEN_PARAM.finditer(chunk):
            for p in m.group(1).split(","):
                p = p.strip().split("=")[0].strip()
                if re.fullmatch(r"[A-Za-z_$][\w$]*(?:-[A-Za-z_$][\w$]*)+", p):
                    names.add(p)
    return names


def repair_js(code: str) -> tuple[str, list[str]]:
    """Rename hyphenated identifiers to camelCase in code (never in strings)."""
    names = _declared_hyphen_names(code)
    if not names:
        return code, []
    rx = re.compile(r"(?<![\w$.\-])(" + "|".join(re.escape(x) for x in sorted(names, key=len, reverse=True))
                    + r")(?![\w$\-])")
    out, last = [], 0
    for a, b in _code_spans(code):
        out.append(code[last:a])
        out.append(rx.sub(lambda m: _camel(m.group(1)), code[a:b]))
        last = b
    out.append(code[last:])
    shown = ", ".join(f"{x} → {_camel(x)}" for x in sorted(names)[:4])
    return "".join(out), [f"renamed hyphenated JavaScript names ({shown}{', …' if len(names) > 4 else ''})"]


# ------------------------------------------------------------- explanations
_HINTS = [
    (r"Missing initializer in const|Unexpected token '-'",
     "a variable name probably contains '-' (e.g. chat-history). JavaScript names can't have hyphens — use chatHistory."),
    (r"Unexpected end of input|was never closed|unexpected EOF",
     "the code stops early — a '{', '(' or '[' is never closed. Finish the cut-off part."),
    (r"Identifier '(\w+)' has already been declared",
     "the same name is declared twice — keep one declaration and delete the other."),
    (r"Unexpected identifier",
     "two things are next to each other without an operator or comma between them, or a keyword is misspelled."),
    (r"Unexpected token '<'|invalid syntax.*<",
     "HTML tags ended up inside code — remove them; code blocks must contain only code."),
    (r"Unexpected string",
     "a string sits right after another value — a '+' or ',' is missing."),
    (r"Invalid or unexpected token",
     "there is a stray character (smart quote “ ”, backslash, or an unclosed string)."),
    (r"expected an indented block",
     "a line ending in ':' needs an indented body under it."),
    (r"unindent does not match|inconsistent use of tabs",
     "indentation is mixed up — use 4 spaces consistently."),
    (r"unterminated string|EOL while scanning",
     "a string is not closed — add the missing quote."),
    (r"'(\w+)' was never closed",
     "a bracket is opened and never closed."),
]


def explain(message: str) -> str:
    for pattern, hint in _HINTS:
        if re.search(pattern, message, re.I):
            return hint
    return ""


def pinpoint(code: str, line: int, col: int | None = None, caret: str = "") -> str:
    """Show the failing line (and its neighbours) with a pointer."""
    lines = code.splitlines()
    if not 1 <= line <= len(lines):
        return ""
    out = []
    for k in range(max(1, line - 1), min(len(lines), line + 1) + 1):
        mark = ">>" if k == line else "  "
        out.append(f"{mark} {k:>4} | {lines[k - 1]}")
        if k == line:
            pad = " " * (len(f"{mark} {k:>4} | "))
            if caret:
                out.append(pad + caret.rstrip())
            elif col:
                out.append(pad + " " * (col - 1) + "^")
    return "\n".join(out)
