"""Phase 7 + Test 10: metrics, full wiring, and architectural compliance.

The import-graph checks parse every module with ast, so they enforce the
layering on the real code rather than describing it.
"""

import ast
from pathlib import Path

import pytest

from tessera.app import build_system
from tessera.config import Config
from tessera.core.approval import StaticApprover
from tessera.core.interfaces import ModelProvider, TokenCounter
from tessera.core.metrics import InstrumentedModel, MetricsStore, resource_snapshot
from tessera.llm.engine import LlamaCppEngine
from tessera.memory.database import Database
from tessera.memory.embedding_service import HashingEmbedder

from conftest import standard_model

PKG = Path(__file__).resolve().parents[1] / "tessera"


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    rel_pkg = path.relative_to(PKG.parent).with_suffix("").parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(rel_pkg[: len(rel_pkg) - node.level + 1])
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def modules(sub: str):
    return sorted((PKG / sub).rglob("*.py"))


@pytest.mark.parametrize("layer, forbidden", [
    ("core", ("tessera.ui", "tessera.tools", "tessera.llm.engine", "tessera.llm.server_backend", "tessera.skills.engine")),
    ("agents", ("tessera.ui", "tessera.skills", "tessera.execution", "tessera.tools", "tessera.llm.engine")),
    ("llm", ("tessera.ui", "tessera.skills", "tessera.execution", "tessera.tools", "tessera.memory")),
    ("memory", ("tessera.ui", "tessera.skills", "tessera.llm", "tessera.agents", "tessera.tools")),
    ("context", ("tessera.ui", "tessera.skills", "tessera.tools", "tessera.llm.engine")),
    ("skills", ("tessera.ui", "tessera.agents", "tessera.llm.engine", "tessera.tools")),
    ("execution", ("tessera.ui", "tessera.agents", "tessera.skills", "tessera.llm.engine")),
])
def test_layering(layer, forbidden):
    for path in modules(layer):
        bad = [m for m in imports_of(path) if m.startswith(forbidden)]
        assert not bad, f"{path.relative_to(PKG)} imports {bad}"


def test_orchestrator_has_no_skill_names_or_conditionals():
    src = (PKG / "core" / "orchestrator.py").read_text() + (PKG / "agents" / "pipeline.py").read_text()
    for name in ("calculator", "find_files", "read_pdf", "desktop_task", "linkedin", "luna"):
        assert name not in src.lower()


def test_prompts_independent_of_execution():
    assert imports_of(PKG / "llm" / "prompts.py") <= {"__future__", "dataclasses"}


def test_metrics_store_and_instrumented_model():
    db = Database(":memory:")
    store = MetricsStore(db)
    base = standard_model()
    base.reset_calls = 0
    base.reset_context = lambda: setattr(base, "reset_calls", base.reset_calls + 1)
    model = InstrumentedModel(base, store)
    assert isinstance(model, ModelProvider) and isinstance(model, TokenCounter)
    from tessera.core.types import Message
    base.queue = ["two words"]
    model.generate([Message.system("Stage: router.\nx"), Message.user("hi")], max_output_tokens=1)
    model.reset_context()
    assert base.reset_calls == 1
    s = store.summary()
    assert s["llm_latency_ms"]["n"] == 1 and s["llm_hit_length_limit"]["n"] == 1
    assert "router" in store.by_tag("llm_latency_ms", "stage")
    assert "latency[router]" in store.report()
    snap = resource_snapshot()
    assert snap.rss_mb is None or snap.rss_mb > 0


def test_full_system_wiring_one_model(tmp_path):
    cfg = Config(database_path=str(tmp_path / "db.sqlite"), working_directory=str(tmp_path / "w"),
                 skill_directory=str(tmp_path / "g"), checkpoint_directory=str(tmp_path / "ck"), embedder="hashing")
    base = standard_model()
    loads_before = LlamaCppEngine._live_instance
    system = build_system(cfg, base, StaticApprover(), embedder=HashingEmbedder())
    natives = {s.name for s in system.engine.registry.all() if s.origin == "native"}
    assert natives == {"write_python_module", "build_project", "find_chat_app", "chat_with_app",
                       "look_at_screen", "desktop_task", "improve_source"}
    # every consumer holds the same wrapped instance, which wraps the one base model
    pipeline = system.app.orchestrator.pipeline
    models = {id(a._model) for a in pipeline._agents.values()} | {id(system.app.orchestrator._router._model)}
    assert len(models) == 1 and system.app.model._inner is base
    assert LlamaCppEngine._live_instance is loads_before
    reply = system.app.orchestrator.handle("Hello.")
    assert reply.text == "final answer"
    assert system.metrics.summary()["turn_seconds"]["n"] == 1
    assert not system.vision.supported  # scripted model has no image input — reported, not faked
