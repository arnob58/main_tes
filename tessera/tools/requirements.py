"""Requirement checks: the names the user asked for must be in the file.

Real failure: asked for a server that loads Tessera with load_model / build_system /
StaticApprover and answers from orchestrator.handle(), Gemma wrote a plausible server
that imports none of it and "simulates" the reply. It passed syntax, duplicate and import
checks — because nothing checked the *request*.

Deterministic, no model calls:
  extract_requirements(request)  code-like names in the user's message — backticked text,
                                 CamelCase, snake_case, name(...) calls, a.b.c chains — and
                                 URL paths such as "/chat"
  missing_requirements(file, …)  which of those the file never mentions
  fake_code_problems(file)       code that announces it is simulated / dummy / mocked

Python-style names only apply to .py files; URL paths and backticked items apply to all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BACKTICK = re.compile(r"`([^`\n]{2,60})`")
_CALL = re.compile(r"\b([A-Za-z_][\w.]*[A-Za-z_]\w*)\(")   # name( — no space, so prose like "does (" isn't a call
_DOTTED = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b")
_SNAKE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")
_URL_PATH = re.compile(r"""["'`](/[A-Za-z0-9_\-/]*)["'`]""")
_FILENAME = re.compile(r"\.(py|html?|js|css|json|md|txt|toml|ya?ml|csv)$", re.I)
_FAKE = re.compile(r"\b(simulat(?:e|es|ed|ing|ion)|dummy|mock(?:ed|ing|s)?|fake[ds]?|stub(?:bed)?)\b", re.I)

# Words that look like identifiers but are ordinary English or too generic to require.
_IGNORE = {"e.g", "i.e", "etc", "vs", "ok", "JavaScript", "TypeScript", "GitHub", "YouTube", "LinkedIn", "PowerShell",
           "OpenAI", "WebSocket", "ChatGPT", "MacBook", "iPhone"}


@dataclass(frozen=True)
class Requirement:
    token: str
    kind: str   # "path" | "name" | "literal"

    def applies_to(self, suffix: str) -> bool:
        return self.kind in ("path", "literal") or suffix.lower() == ".py"

    def present_in(self, text: str) -> bool:
        if self.kind == "path":
            return re.search(r"""["'`]""" + re.escape(self.token) + r"""["'`/?]""", text) is not None
        if self.kind == "literal":
            return self.token in text
        last = self.token.split(".")[-1]
        return re.search(r"\b" + re.escape(last) + r"\b", text) is not None


def extract_requirements(request: str, limit: int = 12) -> list[Requirement]:
    found: dict[str, Requirement] = {}

    def add(tok: str, kind: str) -> None:
        tok = tok.strip().strip(".")
        if not tok or tok in _IGNORE or _FILENAME.search(tok) or len(found) >= limit:
            return
        if kind == "name" and "." in tok:
            parts = tok.split(".")
            if any(len(p) < 2 for p in parts):        # "e.g", "v1.2" etc.
                return
        found.setdefault(tok, Requirement(tok, kind))

    for m in _URL_PATH.finditer(request):
        if len(m.group(1)) > 1:
            add(m.group(1), "path")
    # File and folder paths are where things go, not names the code must use.
    request = re.sub(r"[^\s\"'`]*[\\/][^\s\"'`]*", " ", request)
    for m in _BACKTICK.finditer(request):
        tok = m.group(1).strip()
        if re.fullmatch(r"[\w.]+", tok):
            add(tok, "name")
    for rx in (_CALL, _DOTTED, _CAMEL, _SNAKE):
        for m in rx.finditer(request):
            add(m.group(1), "name")
    # A dotted chain already implies its last part; drop bare duplicates of it.
    reqs = list(found.values())
    tails = {r.token.split(".")[-1] for r in reqs if "." in r.token}
    return [r for r in reqs if "." in r.token or r.token not in tails]


def missing_requirements(text: str, reqs: list[Requirement], suffix: str) -> list[str]:
    missing = [r.token for r in reqs if r.applies_to(suffix) and not r.present_in(text)]
    if not missing:
        return []
    return [f"Requirement check: the request asks for {', '.join(missing)} but the file never uses "
            f"{'it' if len(missing) == 1 else 'them'}. Implement exactly what was asked — don't substitute a simpler "
            "or simulated version."]


def fake_code_problems(text: str, suffix: str) -> list[str]:
    if suffix.lower() not in (".py", ".js", ".mjs"):
        return []
    hits = sorted({m.group(1).lower() for m in _FAKE.finditer(text)})
    if not hits:
        return []
    return [f"Requirement check: the code describes itself as {', '.join(hits)} — build the real thing, not a "
            "simulation or placeholder."]


def requirement_problems(text: str, request: str, suffix: str) -> list[str]:
    if not request:
        return []
    return missing_requirements(text, extract_requirements(request), suffix) + fake_code_problems(text, suffix)
