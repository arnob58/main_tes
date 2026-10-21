"""Deterministic parsers for agent output conventions.

Kept separate from the agents so the conventions can be tested (and later
changed, e.g. to grammar-constrained JSON) without touching control flow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..core.types import SkillRequest

_ACTION = re.compile(r"^\s*ACTION:\s*([A-Za-z_][\w.\-]*)\s*(.*?)\s*$", re.M)
_FENCE_OPEN = re.compile(r"\A[ \t]*\n?[ \t]*```[^\n`]*\n")
_FENCE_CLOSE = re.compile(r"```[ \t]*$", re.M)  # also accepts a fence glued to the last code line
_VERDICT = re.compile(r"^\s*VERDICT:\s*(PASS|REVISE)\b\s*[-:–—]?\s*(.*)$", re.I | re.M)


@dataclass(frozen=True)
class _Span:
    _end: int

    def end(self) -> int:
        return self._end


def _balanced_object(text: str, start: int) -> tuple[str, int]:
    """Return the {...} starting at `start` (string-aware), or ('', start)."""
    if start >= len(text) or text[start] != "{":
        return "", start
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        i += 1
    return "", start


@dataclass(frozen=True)
class ParsedAction:
    request: SkillRequest | None
    error: str = ""          # set when an ACTION line exists but is malformed
    preamble: str = ""       # draft text that preceded the ACTION line


_FILE_START = re.compile(r"^[ \t]*FILE:[ \t]*(\S[^\n]*?)[ \t]*$", re.M)
_FILE_END = re.compile(r"^[ \t]*END FILE[ \t]*$", re.M)
_JSON_PATH = re.compile(r'"path"\s*:\s*"([^"\n]+)"')
_JSON_CONTENT = re.compile(r'"content"\s*:\s*"', re.S)


def parse_file_block(text: str) -> ParsedAction | None:
    """Plain-text file format — no JSON, no escaping:
        FILE: folder/name.ext
        ...raw content...
        END FILE
    """
    m = _FILE_START.search(text)
    if not m:
        return None
    path = m.group(1).strip().strip("`'\"")
    body = text[m.end():].lstrip("\n")
    end = _FILE_END.search(body)
    preamble = text[: m.start()].strip()
    if end is None:
        return ParsedAction(None, f"The FILE block for {path} has no END FILE line (it was probably cut off), "
                                  "so nothing was saved. For a long file use create_file instead.", preamble)
    content = body[: end.start()]
    fence = re.match(r"\s*```[^\n]*\n(.*?)\n?[ \t]*```\s*\Z", content, re.S)  # tolerate a fence inside
    if fence:
        content = fence.group(1) + "\n"
    return ParsedAction(SkillRequest("write_file", {"path": path, "content": content}), preamble=preamble)


def _lenient_file_args(text: str, start: int) -> dict | None:
    """Gemma's habit: {"path": "...", "content": "<html with "quotes" and \\d>"}.
    Take everything between the opening quote of content and the final `"}`."""
    chunk = text[start:]
    p, c = _JSON_PATH.search(chunk), _JSON_CONTENT.search(chunk)
    if not (p and c):
        return None
    rest = chunk[c.end():]
    close = list(re.finditer(r'"\s*}', rest))
    if not close:
        return None
    raw = rest[: close[-1].start()]
    repl = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/"}
    content = re.sub(r'\\(["\\/nt])', lambda m: repl[m.group(1)], raw)
    return {"path": p.group(1), "content": content}


def parse_action(text: str) -> ParsedAction | None:
    """Return the first ACTION (or FILE block) in `text`, or None if there isn't one."""
    m = _ACTION.search(text)
    fb = _FILE_START.search(text)
    if fb and (not m or fb.start() < m.start()):
        return parse_file_block(text)
    if not m:
        return None
    name, raw_args = m.group(1), m.group(2).strip()
    preamble = text[: m.start()].strip()
    args: dict = {}
    if raw_args:
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            # Small models often spill a JSON object over several lines or put raw
            # newlines inside strings. Try the balanced object, leniently.
            obj, end = _balanced_object(text, m.start(2))
            try:
                args = json.loads(obj, strict=False) if obj else None
            except json.JSONDecodeError:
                args = None
            if args is None:
                args = _lenient_file_args(text, m.start(2))
                if args is not None:
                    return ParsedAction(SkillRequest(name, args), preamble=preamble)
            if args is None:
                return ParsedAction(None, f"Arguments for {name} are not valid JSON ({exc.msg}). Use one line of "
                                          f'JSON, and put long content in a fenced code block after the line, e.g.\n'
                                          f'ACTION: {name} {{"path": "folder/file.html"}}\n```html\n...\n```', preamble)
            m = _Span(end)
        if not isinstance(args, dict):
            return ParsedAction(None, f"Arguments for {name} must be a JSON object.", preamble)

    # A fenced code block right after the ACTION line is passed as `content`.
    # Small models write whole files far more reliably this way than as an
    # escaped one-line JSON string.
    rest = text[m.end():]
    opener = _FENCE_OPEN.match(rest)
    if opener:
        body = rest[opener.end():]
        closer = _FENCE_CLOSE.search(body)
        if closer is None:
            return ParsedAction(None, f"The content block for {name} was never closed (the reply was probably "
                                      "cut off), so nothing was done.", preamble)
        args.setdefault("content", body[: closer.start()])
    return ParsedAction(SkillRequest(name, args), preamble=preamble)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    feedback: str
    parsed: bool


def parse_verdict(text: str) -> Verdict:
    """Last VERDICT line wins. Missing verdict → treat as pass (don't loop on noise)."""
    matches = list(_VERDICT.finditer(text))
    if not matches:
        return Verdict(True, "", False)
    m = matches[-1]
    passed = m.group(1).upper() == "PASS"
    feedback = "" if passed else (m.group(2).strip() or "Address the problems listed above.")
    if not passed:
        # Include the evaluator's listed problems, not just the headline fix.
        body = text[: m.start()].strip()
        if body:
            feedback = f"{feedback}\n{body}"
    return Verdict(passed, feedback, True)
