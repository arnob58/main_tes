"""Phase 1 tool bridge: honest about having no skills.

The pipeline already speaks the ToolBridge protocol, so Phase 3 swaps this
for the real skill engine in the composition root — nothing else changes.
"""

from __future__ import annotations

from ..core.types import Observation, SkillRequest


class NullToolBridge:
    def describe_capabilities(self) -> str:
        return "No skills are installed yet; actions cannot be executed in this version."

    def execute(self, request: SkillRequest) -> Observation:
        return Observation(
            status="unavailable",
            source=request.skill,
            result="No skill engine is installed, so this action was not executed.",
        )
