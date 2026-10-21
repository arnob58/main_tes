"""Patch 10 — from the real web_server.py run: the weak tool never saw the request and
looped 10+ times with reworded errors; chat.html's final reply was empty."""

from tessera.app import build_app, build_system
from tessera.config import Config
from tessera.core.approval import StaticApprover
from tessera.core.types import Observation
from tessera.llm.scripted import ScriptedModel
from tessera.memory.embedding_service import HashingEmbedder


def system(tmp_path, model):
    cfg = Config(database_path=str(tmp_path / "db.sqlite"), working_directory=str(tmp_path / "w"),
                 skill_directory=str(tmp_path / "g"), checkpoint_directory=str(tmp_path / "ck"),
                 output_directory=str(tmp_path / "out"), embedder="hashing", gui_enabled=False)
    return build_system(cfg, model, StaticApprover(True), embedder=HashingEmbedder())


def test_write_python_module_is_create_file_and_sees_request_and_named_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main_ref.py").write_text("def load_model(cfg):\n    return 'REFERENCE BODY'\n")
    model = (ScriptedModel(context_window=32768).when_agent("fileplan", "SECTION: all | server")
             .when_agent("filesection", "def load_model(cfg):\n    return cfg\n"))
    sysm = system(tmp_path, model)
    request = f"Read {tmp_path / 'main_ref.py'} then build web/server.py that uses load_model(cfg)"
    obs = sysm.engine.registry.native("write_python_module").invoke(
        {"path": "web/server.py", "description": "a server"}, task_id="t", purpose=request)
    assert obs.status == "success", obs.summary()
    section_prompt = [c for c in model.calls if c.messages[0].content.startswith("Stage: filesection.")][0]
    body = section_prompt.messages[-1].content
    assert "User's request" in body and "load_model(cfg)" in body          # the writer saw the request
    assert "REFERENCE BODY" in body                                        # …and the file the user named


def test_empty_final_reply_is_replaced_with_a_factual_summary(tmp_path):
    class Saver:
        def describe_capabilities(self):
            return "save(x)"

        def execute(self, req):
            p = tmp_path / "chat.html"
            p.write_text("<p>hi</p>")
            return Observation("success", "save", artifacts=(str(p),))

    acts = iter(['ACTION: save {"path": "chat.html"}', "done"])
    model = (ScriptedModel().when_agent("router", "Medium|yes").when_agent("interpret", "i")
             .when_agent("develop", lambda m: next(acts)).when_agent("respond", ""))
    reply = build_app(Config(), model, bridge=Saver()).orchestrator.handle("build chat.html")
    assert reply.text == f"Saved: {tmp_path / 'chat.html'}"


def test_empty_reply_with_nothing_saved_says_so():
    model = ScriptedModel().when_agent("router", "Instant|no").when_agent("interpret", "i").when_agent("respond", "  ")
    reply = build_app(Config(), model).orchestrator.handle("hi")
    assert "wasn't able" in reply.text
