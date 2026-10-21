from tessera.agents.parsing import parse_action, parse_verdict
from tessera.app import build_app
from tessera.config import Config
from tessera.core.task_state import TaskStatus
from tessera.core.types import Message, Observation
from tessera.llm.scripted import ScriptedModel

from conftest import stages_called, standard_model


class RecordingBridge:
    def __init__(self):
        self.requests = []

    def describe_capabilities(self):
        return "add(a, b): adds numbers"

    def execute(self, request):
        self.requests.append(request)
        return Observation("success", request.skill, result=request.arguments["a"] + request.arguments["b"])


def test_each_complexity_runs_its_stages(cfg):
    for route, expected in [
        ("Instant|no", ["router", "interpret", "respond"]),
        ("Medium|no", ["router", "interpret", "develop", "respond"]),
        ("Complex|no", ["router", "interpret", "develop", "evaluate", "respond"]),
    ]:
        model = standard_model(route)
        reply = build_app(cfg, model).orchestrator.handle("hello")
        assert stages_called(model) == expected
        assert reply.text == "final answer" and reply.status is TaskStatus.COMPLETED


def test_handoffs_flow_forward_and_history_is_scoped(cfg):
    model = standard_model("Complex|no")
    app = build_app(cfg, model)
    app.orchestrator.context.extend([Message.user("earlier q"), Message.assistant("earlier a")])
    app.orchestrator.handle("new question")
    by_stage = {s: c for s, c in zip(stages_called(model), model.calls)}

    def contents(stage):
        return [m.content for m in by_stage[stage].messages]

    assert "earlier q" in contents("interpret")          # interpret sees history
    assert "earlier q" not in contents("develop")        # develop works from notes
    assert "INTENT: greet" in contents("develop")[-1]    # analysis handed forward
    assert "draft-v1" in contents("evaluate")[-1]
    assert "draft-v1" in contents("respond")[-1]


def test_revision_loop_is_bounded(cfg):
    model = standard_model("Complex|no", verdicts=["VERDICT: REVISE - fix it"])  # always revise
    reply = build_app(cfg.with_overrides(max_revisions=2), model).orchestrator.handle("q")
    assert stages_called(model).count("develop") == 3
    assert reply.task["counters"]["revisions"] == 2
    revised_develop = [c for s, c in zip(stages_called(model), model.calls) if s == "develop"][1]
    assert "REVISION REQUESTED:\nfix it" in revised_develop.messages[-1].content


def test_develop_skill_request_round_trip(cfg):
    develop_replies = iter(['Need math.\nACTION: add {"a": 2, "b": 3}', "The sum is 5."])
    model = (
        standard_model("Complex|yes")
        .when_agent("develop", lambda m: next(develop_replies))
    )
    model.rules.insert(0, model.rules.pop())  # make the new develop rule win
    bridge = RecordingBridge()
    reply = build_app(cfg, model, bridge=bridge).orchestrator.handle("what is 2+3")

    assert [r.skill for r in bridge.requests] == ["add"]
    stages = stages_called(model)
    assert stages == ["router", "interpret", "develop", "develop", "evaluate", "respond"]
    second_develop = model.calls[3].messages[-1].content
    assert "OBSERVATIONS:" in second_develop and "result: 5" in second_develop
    assert "AVAILABLE CAPABILITIES:\nadd(a, b)" in model.calls[2].messages[-1].content
    assert "result: 5" in model.calls[-1].messages[-1].content  # respond grounded in evidence
    assert reply.task["counters"]["skill_calls"] == 1


def test_malformed_action_becomes_error_observation(cfg):
    replies = iter(["ACTION: add {not json", "gave up"])
    model = standard_model("Medium|yes").when_agent("develop", lambda m: next(replies))
    model.rules.insert(0, model.rules.pop())
    bridge = RecordingBridge()
    build_app(cfg, model, bridge=bridge).orchestrator.handle("q")
    assert bridge.requests == []
    assert "not valid JSON" in model.calls[3].messages[-1].content


def test_null_bridge_reports_unavailable_honestly(cfg):
    replies = iter(['ACTION: read_file {"path": "x"}', "Could not read it."])
    model = standard_model("Medium|yes").when_agent("develop", lambda m: next(replies))
    model.rules.insert(0, model.rules.pop())
    build_app(cfg, model).orchestrator.handle("read x")
    assert "status=unavailable" in model.calls[3].messages[-1].content


def test_pass_budget_always_leaves_room_for_respond():
    cfg = Config(max_agent_iterations=3, max_skill_calls=99)
    model = standard_model("Complex|yes").when_agent("develop", 'ACTION: loop {}')
    model.rules.insert(0, model.rules.pop())
    reply = build_app(cfg, model).orchestrator.handle("q")
    stages = stages_called(model)
    assert stages[-1] == "respond" and len(stages) - 1 == 3  # minus router
    assert reply.status is TaskStatus.COMPLETED


def test_parsers():
    assert parse_action("no action here") is None
    a = parse_action('thinking\nACTION: read_pdf {"path": "a.pdf"}\n')
    assert a.request.skill == "read_pdf" and a.request.arguments == {"path": "a.pdf"} and a.preamble == "thinking"
    assert parse_action("ACTION: list_dir").request.arguments == {}
    assert parse_action('ACTION: x ["list"]').error
    v = parse_verdict("1. wrong unit\nVERDICT: REVISE - use metres")
    assert not v.passed and v.feedback.startswith("use metres") and "wrong unit" in v.feedback
    assert parse_verdict("looks fine\nverdict: pass").passed
    assert parse_verdict("no verdict line").parsed is False


def test_bad_pipeline_config_rejected():
    import pytest
    with pytest.raises(ValueError, match="unknown agents"):
        build_app(Config(pipelines={"instant": ["dream", "respond"]}), ScriptedModel())
    with pytest.raises(ValueError, match="must end"):
        build_app(Config(pipelines={"instant": ["interpret"]}), ScriptedModel())
