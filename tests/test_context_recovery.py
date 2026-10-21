"""Phase 4: token tracking, AST analysis, fresh-context continuation, project builds.

The simulated model knows a target module and answers continuation prompts
*only* from what the prompt contains (the incomplete section / source tail),
so these tests prove the externalised state is sufficient to resume.
"""

import json
import re

import pytest

from tessera.agents.agent import Agent, AgentInput, PromptTooLarge
from tessera.config import Config
from tessera.context.checkpoint import CheckpointStore
from tessera.context.continuation import merge_continuation, strip_fences
from tessera.context.project_builder import ProjectBuilder, build_order, parse_plan, PlanError
from tessera.context.recovery import LongCodeGenerator
from tessera.context.source_analyzer import analyze, split_complete
from tessera.context.tracker import ContextTracker
from tessera.core.types import Message, Observation
from tessera.llm.prompts import DEFAULT_SPECS
from tessera.llm.scripted import ScriptedModel

TARGET = "import math\n\n" + "\n\n".join(
    f'def func_{i}(x: float, scale: float = {i}.0) -> float:\n'
    f'    """Compute variant {i} of the transform."""\n'
    f'    y = math.sqrt(abs(x)) * scale + {i}\n'
    f'    return round(y, {i % 4})\n'
    for i in range(30)
) + "\n\nclass Registry:\n    def __init__(self):\n        self.items = [func_0, func_1]\n\n    def apply(self, x):\n        return [f(x) for f in self.items]\n"


def _section(prompt: str, title: str) -> str:
    m = re.search(rf"{title}[^\n]*:\n(.*?)(?:\n\n[A-Z][A-Z ]+(?: \(|:)|\Z)", prompt, re.S)
    return m.group(1) if m else ""


class SimulatedCoder(ScriptedModel):
    """Writes TARGET; resumes using only the prompt's externalised state."""

    def __init__(self, target=TARGET, window=320, fail_on_round=None):
        super().__init__(context_window=window)
        self.target = target
        self.resets = 0
        self.fail_on_round = fail_on_round
        self.when_agent("codewriter", self.target)
        self.when_agent("codecontinue", self._continue)

    def generate(self, messages, **kw):
        if self.fail_on_round is not None and len(self.calls) + 1 == self.fail_on_round:
            self.calls.append(None)
            raise RuntimeError("simulated crash")
        return super().generate(messages, **kw)

    def reset_context(self):
        self.resets += 1

    def _continue(self, messages):
        prompt = messages[-1].content
        incomplete = _section(prompt, "CURRENT INCOMPLETE SECTION")
        if incomplete and incomplete != "(empty)":
            pos = self.target.find(incomplete)
            assert pos >= 0, "incomplete section must be verbatim target text"
            return self.target[pos + len(incomplete):]
        tail = _section(prompt, "RELEVANT SOURCE CODE").rstrip("\n").splitlines()[-2:]
        anchor = "\n".join(tail)
        pos = self.target.find(anchor)
        assert pos >= 0
        return self.target[pos + len(anchor):]


# ----------------------------------------------------------------- tracker
def test_tracker_threshold_math():
    model = ScriptedModel(context_window=100)
    t = ContextTracker(model, 0.9)
    u = t.measure([Message.user(" ".join(["w"] * 76))], reserved_output=10)  # 76 + 4 = 80 prompt
    assert u.prompt_tokens == 80 and u.ratio == 0.9 and u.needs_recovery
    assert u.safe_output_budget == 10
    assert not t.measure([Message.user("hi")], reserved_output=10).needs_recovery


# --------------------------------------------------------- source analysis
def test_split_never_drops_text():
    src = TARGET[: len(TARGET) // 2]  # cut mid-function
    prefix, suffix, err = split_complete(src)
    assert prefix + suffix == src and err
    compile(prefix, "p", "exec")
    assert suffix.startswith("def func_")


def test_decorator_stays_with_function():
    src = "import x\n\n@decorate\ndef f(\n    a,"
    prefix, suffix, _ = split_complete(src)
    assert prefix == "import x\n\n" and suffix.startswith("@decorate")


def test_suspect_last_unit_moves_to_suffix():
    src = "def a():\n    return 1\n\ndef b():\n    x = 1\n"
    state = analyze(src, last_unit_suspect=True)
    assert state.incomplete_suffix == "def b():\n    x = 1\n"
    assert [f.name for f in state.functions] == ["a"]


def test_interfaces_extracted():
    state = analyze(TARGET)
    text = state.interfaces_text()
    assert "import math" in text and "def func_3(x: float, scale: float=3.0) -> float" in text
    assert "class Registry:" in text and "    def apply(self, x)" in text
    assert state.is_complete


def test_merge_removes_repeated_overlap():
    a = "def f():\n    return 1\n\ndef g():\n    x = 2\n"
    b = "def g():\n    x = 2\n    return x\n"
    assert merge_continuation(a, b) == "def f():\n    return 1\n\ndef g():\n    x = 2\n    return x\n"
    assert merge_continuation("x = 1\ny = ", "y = 2\n") == "x = 1\ny = 2\n"
    assert merge_continuation("x = 1\ny = ", "2\n") == "x = 1\ny = 2\n"
    assert strip_fences("```python\nprint(1)\n```") == "print(1)\n"


# ------------------------------------------------ recovery (Test 6)
def test_generation_larger_than_context_recovers_exactly(tmp_path):
    model = SimulatedCoder(window=320)
    tracker = ContextTracker(model, 0.9)
    store = CheckpointStore(tmp_path / "ck")
    events = []
    gen = LongCodeGenerator(model, tracker, store, events=type("E", (), {"emit": lambda s, c, m: events.append(m)})())
    target_words = len(TARGET.split())
    assert target_words > model.context_window  # genuinely larger than one window

    res = gen.generate("Write the transforms module", tmp_path / "out" / "transforms.py", key="t1")

    assert res.valid and res.source == TARGET                     # integrated exactly, nothing lost or repeated
    assert (tmp_path / "out" / "transforms.py").read_text() == TARGET
    assert res.recoveries >= 2 and model.resets == res.recoveries   # fresh context each time
    for r in res.rounds:                                            # never crossed 90%
        assert r.prompt_tokens + r.output_budget <= 0.9 * 320
    continuation_calls = [c for c in model.calls[1:]]
    for c in continuation_calls:
        assert c.messages[0].content.startswith("Stage: codecontinue.")
        assert "Stage: codewriter." not in " ".join(m.content for m in c.messages)  # no old transcript
        assert "EXISTING INTERFACES" in c.messages[1].content
    assert store.load("t1") is None                                 # checkpoint discarded on success
    assert any("context used" in e for e in events)


def test_resume_after_crash_from_checkpoint(tmp_path):
    store = CheckpointStore(tmp_path / "ck")
    path = tmp_path / "transforms.py"
    crashing = SimulatedCoder(window=320, fail_on_round=2)
    gen = LongCodeGenerator(crashing, ContextTracker(crashing, 0.9), store)
    with pytest.raises(RuntimeError, match="simulated crash"):
        gen.generate("Write the transforms module", path, key="t2")
    saved = store.load("t2")
    assert saved and saved["incomplete_suffix"] and path.exists()
    partial = path.read_text()

    fresh = SimulatedCoder(window=320)  # new process, new model, same disk
    res = LongCodeGenerator(fresh, ContextTracker(fresh, 0.9), store).generate("Write the transforms module", path, key="t2")
    assert res.valid and res.source == TARGET and res.source.startswith(partial)
    assert fresh.calls[0].messages[0].content.startswith("Stage: codecontinue.")


def test_impossible_prompt_fails_loudly(tmp_path):
    model = SimulatedCoder(window=40)
    gen = LongCodeGenerator(model, ContextTracker(model, 0.9), CheckpointStore(tmp_path))
    from tessera.context.recovery import RecoveryFailed
    with pytest.raises(RecoveryFailed):
        gen.generate("x " * 60, tmp_path / "f.py", key="k")


# ------------------------------------------------- agent prompt fitting
def test_agent_shrinks_low_priority_material_to_fit():
    model = ScriptedModel(context_window=400)
    tracker = ContextTracker(model, 0.9)
    agent = Agent(DEFAULT_SPECS["develop"], model, Config(agent_max_tokens=100), tracker)
    big = " ".join(f"m{i}" for i in range(400))
    inp = AgentInput(
        request="Summarise the observation.",
        memory=big,
        observations=(Observation("success", "read", result=" ".join(f"o{i}" for i in range(300))),),
        analysis="INTENT: summarise",
    )
    msgs = agent.build_messages(inp)
    assert tracker.measure(msgs, 100).ratio < 0.9
    body = msgs[-1].content
    assert "Summarise the observation." in body and "INTENT: summarise" in body
    assert "omitted to fit the context window" in body
    assert agent.last_fit["shrinks"] >= 1


def test_agent_raises_when_request_alone_cannot_fit():
    model = ScriptedModel(context_window=60)
    agent = Agent(DEFAULT_SPECS["respond"], model, Config(), ContextTracker(model, 0.9))
    with pytest.raises(PromptTooLarge):
        agent.build_messages(AgentInput(request="word " * 80))


# ------------------------------------------------------- project builder
PLAN = {
    "modules": [
        {"path": "calc/core.py", "purpose": "math helpers", "interfaces": ["def add(a, b)"], "depends_on": []},
        {"path": "calc/cli.py", "purpose": "cli", "interfaces": ["def main(argv)"], "depends_on": ["calc/core.py"]},
    ],
    "tests": [{"path": "tests/test_core.py", "purpose": "tests", "depends_on": ["calc/core.py"]}],
}
FILES = {
    "calc/core.py": "def add(a, b):\n    return a + b\n",
    "calc/cli.py": "from calc.core import add\n\n\ndef main(argv):\n    return add(int(argv[0]), int(argv[1]))\n",
    "tests/test_core.py": "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parents[1]))\nfrom calc.core import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
}


def project_model():
    m = ScriptedModel(context_window=4096)
    m.when_agent("projectplan", "Plan:\n" + json.dumps(PLAN))

    def write(messages):
        path = re.search(r"Write the file (\S+)", messages[-1].content).group(1)
        return FILES[path]

    m.when_agent("codewriter", write)
    return m


def test_project_build_and_integration(tmp_path):
    (tmp_path / "proj" / "calc").mkdir(parents=True)
    (tmp_path / "proj" / "calc" / "__init__.py").write_text("")
    model = project_model()
    store = CheckpointStore(tmp_path / "ck")
    builder = ProjectBuilder(model, LongCodeGenerator(model, ContextTracker(model, 0.9), store))
    report = builder.build("A tiny calculator", tmp_path / "proj", key="p")
    assert report.ok, report.summary() + report.integration_output
    assert [m.path for m in report.modules] == ["calc/core.py", "calc/cli.py", "tests/test_core.py"]
    cli_prompt = [c for c in model.calls if "calc/cli.py" in c.messages[-1].content][0].messages[-1].content
    assert "def add(a, b)" in cli_prompt and "return a + b" not in cli_prompt  # interfaces, not code

    n = len(model.calls)
    again = builder.build("A tiny calculator", tmp_path / "proj", key="p")
    assert {m.status for m in again.modules} == {"skipped"} and len(model.calls) == n


def test_plan_validation():
    with pytest.raises(PlanError, match="unsafe"):
        parse_plan('{"modules": [{"path": "../evil.py"}]}')
    with pytest.raises(PlanError):
        parse_plan("no json here")
    specs = parse_plan(json.dumps(PLAN))
    assert [s.path for s in build_order(list(reversed(specs)))][:2] == ["calc/core.py", "calc/cli.py"]
