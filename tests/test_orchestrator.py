import io

from tessera.app import build_app
from tessera.core.task_state import CancellationToken, TaskStatus
from tessera.core.types import Role
from tessera.llm.prompts import AgentSpec, DEFAULT_SPECS
from tessera.llm.scripted import ScriptedModel
from tessera.ui.console import ConsoleUI

from conftest import stages_called, standard_model


def test_hello_end_to_end(cfg):
    model = standard_model("Instant|no")
    app = build_app(cfg, model)
    reply = app.orchestrator.handle("Hello.")
    assert reply.status is TaskStatus.COMPLETED
    assert reply.route.label() == "Instant|no"
    assert reply.task["counters"]["skill_calls"] == 0
    assert reply.model_calls == 3  # router + interpret + respond
    roles = [(m.role, m.content) for m in app.orchestrator.context.messages()]
    assert roles == [(Role.USER, "Hello."), (Role.ASSISTANT, "final answer")]


def test_internal_state_never_enters_conversation(cfg):
    model = standard_model("Complex|no")
    app = build_app(cfg, model)
    app.orchestrator.handle("q")
    contents = " ".join(m.content for m in app.orchestrator.context.messages())
    assert "INTENT" not in contents and "draft-v1" not in contents and "VERDICT" not in contents


def test_events_are_emitted(cfg):
    seen = []
    app = build_app(cfg, standard_model("Medium|no"))
    app.events.subscribe(lambda ch, msg: seen.append(ch))
    app.orchestrator.handle("q")
    assert "Router" in seen and "Agent" in seen


def test_cancellation_between_passes(cfg):
    token = CancellationToken()
    model = standard_model("Complex|no")
    app = build_app(cfg, model)
    app.events.subscribe(lambda ch, msg: token.cancel() if msg == "develop…" else None)
    reply = app.orchestrator.handle("q", token)
    assert reply.status is TaskStatus.CANCELLED
    assert "develop" not in stages_called(model)  # stopped before the model call
    assert len(app.orchestrator.context) == 0


def test_model_failure_reported_not_raised(cfg):
    class Broken(ScriptedModel):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    reply = build_app(cfg, Broken()).orchestrator.handle("q")
    assert reply.status is TaskStatus.FAILED and "CUDA out of memory" in reply.text


def test_all_agents_share_one_model_instance(cfg):
    model = standard_model()
    app = build_app(cfg, model)
    pipeline = app.orchestrator._pipeline
    assert all(agent._model is model for agent in pipeline._agents.values())
    assert app.orchestrator._router._model is model


def test_new_stage_needs_no_core_changes(cfg):
    """OCP check: add a 'critic' stage purely via specs + config."""
    specs = dict(DEFAULT_SPECS)
    specs["critic"] = AgentSpec("critic", "Stage: critic.\nPoint out one weakness.")
    c = cfg.with_overrides(pipelines={**cfg.pipelines, "medium": ["interpret", "develop", "critic", "respond"]})
    model = standard_model("Medium|no").when_agent("critic", "weak spot")
    build_app(c, model, specs=specs).orchestrator.handle("q")
    assert stages_called(model) == ["router", "interpret", "develop", "critic", "respond"]


def test_console_session(cfg):
    inputs = iter(["Hello.", "/history", "/quit"])
    out = io.StringIO()
    app = build_app(cfg, standard_model())
    ConsoleUI(app.orchestrator, app.events, out=out, input_fn=lambda _: next(inputs)).loop()
    text = out.getvalue()
    assert "[Router] Instant|no" in text
    assert "Tessera:\nfinal answer" in text
    assert "user: Hello." in text and text.rstrip().endswith("Bye.")
