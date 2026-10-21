"""One generic logical agent, configured entirely by an AgentSpec.

Why one class instead of interpret.py / develop.py / …: the stages differ
only in prompt, budget and what material they see. Encoding that as data
keeps them domain-general and makes new stages a config change (OCP). The
agent depends on the ModelProvider protocol, not on llama.cpp (DIP).

Each call builds a fresh message list — no agent inherits another's prompt.

Token budgeting (Phase 4): when a ContextTracker is supplied, the prompt is
measured with the real tokenizer and, if prompt + output would cross the
recovery threshold, the least important material is shrunk first (memory,
old history, old observations…) until it fits. The request itself is never cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import Config
from ..context.tracker import ContextTracker
from ..core.interfaces import ModelProvider
from ..core.types import GenerationResult, Message, Observation
from ..llm.prompts import AgentSpec


class PromptTooLarge(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentInput:
    request: str
    history: Sequence[Message] = ()
    analysis: str = ""
    draft: str = ""
    evaluation_feedback: str = ""
    observations: Sequence[Observation] = ()
    capabilities: str = ""
    memory: str = ""


# Hard upper bounds that keep absurd inputs out of the tokenizer; real
# fitting is token-based below.
SECTION_CHAR_CAP = 12000
OBSERVATION_CHAR_CAP = 4000
MIN_OUTPUT_TOKENS = 32

# Shrink order: first entry is sacrificed first.
SHRINK_ORDER = ("memory", "history", "observations", "capabilities", "analysis", "draft", "feedback")


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + "\n…[truncated]"


def shrink(text: str) -> str:
    """Keep the head and tail, drop the middle — tails often hold the latest result."""
    if len(text) < 200:
        return ""
    keep = len(text) // 4
    return f"{text[:keep]}\n…[{len(text) - 2 * keep} chars omitted to fit the context window]…\n{text[-keep:]}"


class Agent:
    def __init__(self, spec: AgentSpec, model: ModelProvider, cfg: Config, tracker: ContextTracker | None = None) -> None:
        self.spec = spec
        self._model = model
        self._tracker = tracker
        self._max_tokens = int(getattr(cfg, spec.budget_key))
        self._temperature = spec.temperature if spec.temperature is not None else cfg.default_temperature
        self.last_fit: dict = {}

    @property
    def name(self) -> str:
        return self.spec.name

    # -------------------------------------------------------------- sections
    def _sections(self, inp: AgentInput) -> dict[str, str]:
        s: dict[str, str] = {}
        name = self.spec.name
        if inp.memory and self.spec.sees_memory:
            s["memory"] = _cap(inp.memory, SECTION_CHAR_CAP)
        if inp.analysis and name != "interpret":
            s["analysis"] = _cap(inp.analysis, SECTION_CHAR_CAP)
        if inp.capabilities and name == "develop":
            s["capabilities"] = _cap(inp.capabilities, SECTION_CHAR_CAP)
        if inp.observations:
            s["observations"] = _cap("\n\n".join(o.summary(OBSERVATION_CHAR_CAP) for o in inp.observations), SECTION_CHAR_CAP)
        if inp.draft:
            s["draft"] = _cap(inp.draft, SECTION_CHAR_CAP)
        if inp.evaluation_feedback and name == "develop":
            s["feedback"] = _cap(inp.evaluation_feedback, 2000)
        return s

    _TITLES = {
        "memory": "MEMORY", "analysis": "ANALYSIS NOTES", "capabilities": "AVAILABLE CAPABILITIES",
        "observations": "OBSERVATIONS", "draft": "DRAFT", "feedback": "REVISION REQUESTED",
    }
    _ORDER = ("memory", "analysis", "capabilities", "observations", "draft", "feedback")

    def _compose(self, request: str, history: Sequence[Message], sections: dict[str, str]) -> list[Message]:
        msgs = [Message.system(self.spec.system_prompt)]
        if self.spec.sees_history:
            msgs.extend(history)
        parts = [f"REQUEST:\n{request}"]
        parts += [f"{self._TITLES[k]}:\n{sections[k]}" for k in self._ORDER if sections.get(k)]
        msgs.append(Message.user("\n\n".join(parts)))
        return msgs

    def build_messages(self, inp: AgentInput) -> list[Message]:
        return self._fit(inp)[0]

    def _fit(self, inp: AgentInput) -> tuple[list[Message], int]:
        sections = self._sections(inp)
        history = list(inp.history)
        msgs = self._compose(inp.request, history, sections)
        if self._tracker is None:
            return msgs, self._max_tokens

        shrinks = 0
        usage = self._tracker.measure(msgs, self._max_tokens)
        while usage.needs_recovery:
            target = next(
                (k for k in SHRINK_ORDER if (k == "history" and history and self.spec.sees_history) or sections.get(k)),
                None,
            )
            if target is None:
                break
            if target == "history":
                history.pop(0)  # oldest first
            else:
                sections[target] = shrink(sections[target])
            shrinks += 1
            msgs = self._compose(inp.request, history, sections)
            usage = self._tracker.measure(msgs, self._max_tokens)

        budget = self._max_tokens
        if usage.needs_recovery:  # only the request is left: give it the room that remains
            budget = usage.safe_output_budget
            if budget < MIN_OUTPUT_TOKENS:
                raise PromptTooLarge(
                    f"{self.name}: request alone needs {usage.prompt_tokens} of {usage.window} tokens"
                )
        self.last_fit = {"prompt_tokens": usage.prompt_tokens, "shrinks": shrinks, "output_budget": budget}
        return msgs, budget

    def run(self, inp: AgentInput) -> GenerationResult:
        msgs, budget = self._fit(inp)
        return self._model.generate(msgs, max_output_tokens=budget, temperature=self._temperature)
