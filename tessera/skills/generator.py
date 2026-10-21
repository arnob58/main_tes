"""Level 3: asks the shared model to write or repair a skill module.

SRP: prompt construction, code extraction and static contract validation.
It never executes anything. Uses the same ModelProvider as every agent — no
second model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.interfaces import ModelProvider
from ..core.types import Message, SkillRequest
from .contract import ContractError, SkillSpec, parse_skill_source

GENERATE_PROMPT = """Stage: skillgen.
You write one self-contained Python 3 module that performs a capability. Rules:
- Define SKILL = {"name": ..., "description": ..., "parameters": {"arg": "type - meaning"}, "dependencies": [...]}
- Define def run(**kwargs) that reads its arguments from kwargs, does the work, and RETURNS a JSON-serialisable result (dict/list/str/number).
- Prefer the standard library. List any third-party pip package in "dependencies" and import it normally.
- Expand "~" with os.path.expanduser. Never call input(). Don't delete or overwrite user files unless the task explicitly asks.
- Raise an exception with a clear message on failure; don't return fake data.
- Keep it short. Output only the code in one ```python block."""

REPAIR_PROMPT = """Stage: skillfix.
A skill module failed. Fix the cause with the smallest change and return the complete corrected module in one ```python block. Keep the same SKILL name and run(**kwargs) contract."""

_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedSkill:
    source: str
    spec: SkillSpec
    attempts: int


def extract_code(text: str) -> str:
    blocks = _BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "def run" in text:  # model forgot the fence
        return text.strip() + "\n"
    raise GenerationError("no code block in model output")


class SkillGenerator:
    def __init__(self, model: ModelProvider, *, max_tokens: int = 900, attempts: int = 2, temperature: float = 0.2) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._attempts = attempts
        self._temperature = temperature

    def generate(self, request: SkillRequest, name: str) -> GeneratedSkill:
        user = (
            f"Capability name: {name}\n"
            f"Arguments it will be called with: {request.arguments}\n"
            f"Overall task (context only): {request.purpose or 'n/a'}"
        )
        return self._ask(GENERATE_PROMPT, user, name)

    def repair(self, source: str, error: str, name: str, arguments: dict) -> GeneratedSkill:
        user = f"MODULE:\n```python\n{source}```\n\nCALLED WITH: {arguments}\n\nFAILURE:\n{error[-1500:]}"
        return self._ask(REPAIR_PROMPT, user, name)

    def _ask(self, system: str, user: str, name: str) -> GeneratedSkill:
        last_error = ""
        for attempt in range(1, self._attempts + 1):
            prompt = user if not last_error else f"{user}\n\nYour previous answer was invalid: {last_error}. Try again."
            result = self._model.generate(
                [Message.system(system), Message.user(prompt)],
                max_output_tokens=self._max_tokens, temperature=self._temperature,
            )
            try:
                code = extract_code(result.text)
                spec = parse_skill_source(code, default_name=name)
                if result.finish_reason == "length":
                    raise ContractError("output was cut off at the token limit")
                return GeneratedSkill(code, spec, attempt)
            except (GenerationError, ContractError) as exc:
                last_error = str(exc)
        raise GenerationError(f"could not produce a valid skill: {last_error}")
