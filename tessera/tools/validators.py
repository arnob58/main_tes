"""Deterministic checks for files the model writes — so bugs are found and fixed, not shipped.

SRP: given (file name, text) return a list of concrete problems ("" list = ok).
No model calls. Uses `node --check` for JavaScript when Node.js is installed,
otherwise a bracket-balance check that still catches cut-off / duplicated code.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

# Elements whose missing end tag really breaks a page (optional-close tags like
# <p>, <li>, <td> are deliberately not checked to avoid false alarms).
STRICT_CLOSE = {"html", "head", "body", "script", "style", "table", "form", "div", "select", "textarea", "ul", "ol"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


def bracket_problems(code: str, label: str) -> list[str]:
    """Balance (), [], {} outside strings and comments (JS/CSS-ish lexing)."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[tuple[str, int]] = []
    i, line, n = 0, 1, len(code)
    while i < n:
        c = code[i]
        if c == "\n":
            line += 1
        elif c in "\"'`":
            q, i = c, i + 1
            while i < n and code[i] != q:
                if code[i] == "\\":
                    i += 1
                elif code[i] == "\n":
                    line += 1
                    if q != "`":
                        break
                i += 1
        elif code.startswith("//", i) and label != "CSS":
            while i < n and code[i] != "\n":
                i += 1
            continue
        elif code.startswith("/*", i):
            end = code.find("*/", i + 2)
            line += code[i:end if end >= 0 else n].count("\n")
            i = n if end < 0 else end + 2
            continue
        elif c in "([{":
            stack.append((c, line))
        elif c in ")]}":
            if not stack or stack[-1][0] != pairs[c]:
                return [f"{label}: unexpected '{c}' at line {line}"]
            stack.pop()
        i += 1
    if stack:
        ch, ln = stack[-1]
        return [f"{label}: '{ch}' opened at line {ln} is never closed (code cut off or duplicated?)"]
    return []


def _detailed(problem: str, code: str, where, err=None, col=None) -> str:
    """Append the failing line with a pointer and a plain-English hint."""
    from .js_repair import explain, pinpoint

    caret = ""
    if err:
        for k, l in enumerate(err):
            if set(l.strip()) <= {"^"} and l.strip() and k > 0:
                caret = l
                break
    try:
        shown = pinpoint(code, int(where), col, caret)
    except (TypeError, ValueError):
        shown = ""
    hint = explain(problem)
    return problem + (f"\n{shown}" if shown else "") + (f"\nLikely cause: {hint}" if hint else "")


def js_problems(code: str, label: str = "JavaScript") -> list[str]:
    node = shutil.which("node")
    if node:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
            fh.write(code)
            tmp = fh.name
        try:
            p = subprocess.run([node, "--check", tmp], capture_output=True, text=True, timeout=30)
        finally:
            Path(tmp).unlink(missing_ok=True)
        if p.returncode != 0:
            err = p.stderr.splitlines()
            msg = next((l for l in err if "Error" in l), p.stderr.strip()[-200:])
            where = next((l.split(":")[-1] for l in err if l.startswith(tmp)), "?")
            return [_detailed(f"{label}: {msg.strip()} (line {where})", code, where, err)]
        return []
    return bracket_problems(code, label)


class _HtmlChecker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.problems: list[str] = []
        self.scripts: list[tuple[str, int]] = []
        self.styles: list[tuple[str, int]] = []
        self._capture: tuple[str, int, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        self.stack.append((tag, self.getpos()[0]))
        if tag in ("script", "style") and not dict(attrs).get("src"):
            self._capture = (tag, self.getpos()[0], [])

    def handle_endtag(self, tag):
        if self._capture and self._capture[0] == tag:
            kind, ln, parts = self._capture
            (self.scripts if kind == "script" else self.styles).append(("".join(parts), ln))
            self._capture = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                skipped = [t for t, _ in self.stack[i + 1:] if t in STRICT_CLOSE]
                if skipped:
                    self.problems.append(f"HTML: <{skipped[-1]}> not closed before </{tag}> (line {self.getpos()[0]})")
                del self.stack[i:]
                return

    def handle_data(self, data):
        if self._capture:
            self._capture[2].append(data)


def html_problems(text: str) -> list[str]:
    chk = _HtmlChecker()
    chk.feed(text)
    chk.close()
    problems = list(chk.problems)
    if chk._capture:
        kind, ln, parts = chk._capture
        problems.append(f"HTML: <{kind}> starting at line {ln} is never closed (file cut off?)")
        (chk.scripts if kind == "script" else chk.styles).append(("".join(parts), ln))
    open_strict = [(t, ln) for t, ln in chk.stack if t in STRICT_CLOSE]
    if open_strict and not chk._capture:
        t, ln = open_strict[-1]
        problems.append(f"HTML: <{t}> opened at line {ln} is never closed")
    for code, ln in chk.scripts:
        problems += [f"{p} [script starting at line {ln}]" for p in js_problems(code, "JavaScript")]
    for code, ln in chk.styles:
        problems += [f"{p} [style starting at line {ln}]" for p in bracket_problems(code, "CSS")]
    return problems


def validate_file(name: str, text: str) -> list[str]:
    """Problems that make the file broken. Empty list = no problems found."""
    problems: list[str] = []
    suffix = Path(name).suffix.lower()
    lines = text.splitlines()
    fence = next((i + 1 for i, l in enumerate(lines) if "```" in l), None)
    if fence and suffix not in (".md", ".markdown", ".txt"):
        problems.append(f"stray markdown fence ``` at line {fence} — remove chat formatting from the file")
    if suffix in (".html", ".htm"):
        problems += html_problems(text)
    elif suffix in (".js", ".mjs", ".cjs"):
        problems += js_problems(text)
    elif suffix == ".css":
        problems += bracket_problems(text, "CSS")
    elif suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"JSON: {exc.msg} at line {exc.lineno}")
    elif suffix == ".py":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            problems.append(_detailed(f"Python: {exc.msg} at line {exc.lineno}", text, exc.lineno, col=exc.offset))
        else:
            from .py_checks import mock_problems

            problems += mock_problems(text)
    return problems
