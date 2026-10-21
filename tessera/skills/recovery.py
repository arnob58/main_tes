"""Classifies a failed run into a mechanical fix or an escalation.

SRP: decide *what kind* of recovery fits; the engine carries it out.
Known, mechanical cases are handled without the model. Anything novel is
escalated to the model with the full traceback. Identical repeated failures
stop the loop instead of burning retries.
"""

from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass

from ..execution.process_runner import RunResult


class Action(str, enum.Enum):
    INSTALL = "install"
    RETRY_LONGER = "retry_longer"
    FIX_ARGS = "fix_args"
    REPAIR = "repair"        # escalate to the model
    GIVE_UP = "give_up"


@dataclass(frozen=True)
class Plan:
    action: Action
    detail: str = ""
    fixed_args: dict | None = None


_UNEXPECTED_KW = re.compile(r"got an unexpected keyword argument '(\w+)'")
_MISSING_ARG = re.compile(r"missing \d+ required (?:positional|keyword-only) arguments?: (.+)$")


def failure_signature(run: RunResult) -> str:
    """Stable fingerprint of a failure, ignoring line numbers and addresses."""
    tail = re.sub(r"line \d+|0x[0-9a-f]+", "", (run.traceback or run.message)[-600:])
    return hashlib.sha1(f"{run.error_type}|{tail}".encode()).hexdigest()[:16]


class SkillRecovery:
    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def plan(self, run: RunResult, arguments: dict, *, timeout_retried: bool) -> Plan:
        sig = failure_signature(run)
        self._seen[sig] = self._seen.get(sig, 0) + 1
        if self._seen[sig] > 2:
            return Plan(Action.GIVE_UP, "the same failure repeated; stopping")

        if run.missing_module:
            return Plan(Action.INSTALL, run.missing_module)
        if run.timed_out:
            return Plan(Action.GIVE_UP if timeout_retried else Action.RETRY_LONGER, "timeout")
        if run.error_type == "TypeError":
            m = _UNEXPECTED_KW.search(run.message)
            if m and m.group(1) in arguments:
                fixed = {k: v for k, v in arguments.items() if k != m.group(1)}
                return Plan(Action.FIX_ARGS, f"dropped unexpected argument '{m.group(1)}'", fixed)
            if _MISSING_ARG.search(run.message):
                return Plan(Action.REPAIR, "required argument missing")
        return Plan(Action.REPAIR, run.error_type or "unknown failure")
