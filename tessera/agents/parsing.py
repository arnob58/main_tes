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
_VERDICT = re.compile(r"^\s*VERDICT:\s*(PASS|REVISE)\b\s*[-:–—]?\s*(.*)$", re.I | re.M)


@dataclass(frozen=True)
class ParsedAction:
    request: SkillRequest | None
    error: str = ""          # set when an ACTION line exists but is malformed
    preamble: str = ""       # draft text that preceded the ACTION line


def parse_action(text: str) -> ParsedAction | None:
    """Return the first ACTION in `text`, or None if there isn't one."""
    m = _ACTION.search(text)
    if not m:
        return None
    name, raw_args = m.group(1), m.group(2).strip()
    preamble = text[: m.start()].strip()
    if not raw_args:
        return ParsedAction(SkillRequest(name, {}), preamble=preamble)
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return ParsedAction(None, f"Arguments for {name} are not valid JSON: {exc.msg}", preamble)
    if not isinstance(args, dict):
        return ParsedAction(None, f"Arguments for {name} must be a JSON object.", preamble)
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
