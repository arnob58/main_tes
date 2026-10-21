import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessera.config import Config  # noqa: E402
from tessera.llm.scripted import ScriptedModel  # noqa: E402


@pytest.fixture
def cfg() -> Config:
    return Config()


def standard_model(route: str = "Instant|no", verdicts: list[str] | None = None) -> ScriptedModel:
    """A scripted model whose per-stage replies identify the stage."""
    verdicts = list(verdicts or ["VERDICT: PASS"])

    def evaluate(_):
        return verdicts.pop(0) if len(verdicts) > 1 else verdicts[0]

    return (
        ScriptedModel()
        .when_agent("router", route)
        .when_agent("interpret", "INTENT: greet\nTOOLS: none")
        .when_agent("develop", "draft-v1")
        .when_agent("evaluate", evaluate)
        .when_agent("respond", "final answer")
    )


def stages_called(model: ScriptedModel) -> list[str]:
    out = []
    for call in model.calls:
        first = call.messages[0].content
        out.append(first.split("\n", 1)[0].removeprefix("Stage: ").rstrip("."))
    return out
