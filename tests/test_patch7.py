"""Patch 7 — from real runs:
  * web_server.py / p2p_chat.py: section 1 wrote everything, section 2 redefined a function,
    the fix loop targeted the wrong section 12 times → duplicates now removed by Tessera
  * write_python_module saved to a hidden workspace → user's output folder, path reported
  * "read app.py and __main__.py" was answered from memory without reading → Tessera reads
    named files itself; missing files are reported; reads carry over one turn
"""


from tessera.app import build_app, build_system
from tessera.config import Config
from tessera.context.tracker import ContextTracker
from tessera.core.approval import StaticApprover
from tessera.core.file_refs import FileReferenceReader
from tessera.llm.scripted import ScriptedModel
from tessera.memory.embedding_service import HashingEmbedder
from tessera.tools.dedupe import dedupe_sections
from tessera.tools.file_builder import CreateFileTool, SectionedFileBuilder
from tessera.tools.files import WriteFileTool
from tessera.tools.html_builder import HtmlPageBuilder

from conftest import standard_model

OVERRUN_SETUP = '''import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def tessera_loader():
    return "early copy"


def make_handler(system):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            pass
    return Handler
'''
LOADER = '''def tessera_loader():
    from tessera.config import load_config
    return load_config("tessera.toml")
'''
RUNNER = '''def main():
    system = tessera_loader()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), make_handler(system))
    server.serve_forever()


if __name__ == "__main__":
    main()
'''


class WebServerGemma(ScriptedModel):
    """Reproduces the real run: 'setup' overruns into the loader's job."""

    def __init__(self):
        super().__init__(context_window=32768)
        self.when_agent("fileplan", "SECTION: setup | imports and globals\nSECTION: tessera_loader | load Tessera\n"
                                    "SECTION: server_runner | run the server\nNAMES: tessera_loader, main")
        self.when_agent("filesection", self._section)

    def _section(self, messages):
        body = messages[-1].content
        return OVERRUN_SETUP if "section 1 —" in body else LOADER if "section 2 —" in body else RUNNER


def file_tool(tmp_path, model):
    tracker = ContextTracker(model, 0.9)
    writer = WriteFileTool(tmp_path / "out", StaticApprover(True))
    return CreateFileTool(SectionedFileBuilder(model, tracker), writer, HtmlPageBuilder(model, tracker))


# ------------------------------------------------------------- duplicates
def test_overrun_duplicate_removed_without_model_and_saved(tmp_path):
    model = WebServerGemma()
    obs = file_tool(tmp_path, model).invoke({"path": "tessera_web/web_server.py", "description": "web server"},
                                            task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    src = (tmp_path / "out" / "tessera_web" / "web_server.py").read_text()
    compile(src, "web_server.py", "exec")
    assert src.count("def tessera_loader") == 1 and "load_config" in src and "early copy" not in src
    assert any("kept 'tessera_loader' from section 'tessera_loader'" in n for n in obs.notes)
    section_calls = [c for c in model.calls if c.messages[0].content.startswith("Stage: filesection.")]
    assert len(section_calls) == 3                                   # no fix rounds at all


def test_dedupe_keeps_last_copy_when_no_section_owns_the_name():
    texts, notes = dedupe_sections(["a", "b"], ["def f():\n    return 1\n", "def f():\n    return 2\n"], ".py")
    assert texts[0].strip() == "" and "return 2" in texts[1] and notes


def test_dedupe_javascript_functions_with_tricky_braces():
    a = "function render(x) {\n  const s = '}';\n  if (x) { return s; }\n}\nconst keep = 1;\n"
    b = "function render(x) {\n  return x;\n}\n"
    texts, _ = dedupe_sections(["setup", "render"], [a, b], ".js")
    assert texts[0] == "const keep = 1;\n" and "return x" in texts[1]


def test_other_file_types_untouched():
    texts, notes = dedupe_sections(["a", "b"], [".x { }", ".x { }"], ".css")
    assert texts == [".x { }", ".x { }"] and notes == []


# ------------------------------------------------------------- output folder
def test_codegen_tools_save_to_output_folder(tmp_path):
    cfg = Config(database_path=str(tmp_path / "db.sqlite"), working_directory=str(tmp_path / "w"),
                 skill_directory=str(tmp_path / "g"), checkpoint_directory=str(tmp_path / "ck"),
                 output_directory=str(tmp_path / "out"), embedder="hashing", gui_enabled=False)
    model = (ScriptedModel().when_agent("fileplan", "SECTION: all | the module")
             .when_agent("filesection", "def hello():\n    return 'hi'\n"))   # write_python_module → create_file
    system = build_system(cfg, model, StaticApprover(True), embedder=HashingEmbedder())
    obs = system.engine.registry.native("write_python_module").invoke(
        {"path": "p2p_chat/p2p_chat.py", "description": "x"}, task_id="t", purpose="")
    saved = tmp_path / "out" / "p2p_chat" / "p2p_chat.py"
    assert obs.status == "success" and saved.exists()
    assert any(n.startswith(f"saved to {saved}") for n in obs.notes)


def test_respond_announces_files_instead_of_pasting():
    from tessera.llm.prompts import RESPOND
    assert "do not paste the file's contents" in RESPOND.system_prompt


# ------------------------------------------------------- named files are read
def test_named_files_are_read_before_the_model_and_memory_is_skipped(tmp_path):
    (tmp_path / "app.py").write_text("def build_system(cfg, model, approver):\n    return 'SYSTEM'\n")
    reader = FileReferenceReader(base_dirs=(tmp_path,))
    model = standard_model("Medium|yes")
    app = build_app(Config(), model, file_reader=reader)
    seen = []
    app.events.subscribe(lambda ch, msg: seen.append((ch, msg)))
    app.orchestrator.handle(f"First read the file {tmp_path / 'app.py'} then explain it")
    develop = [c for c in model.calls if c.messages[0].content.startswith("Stage: develop.")][0].messages[-1].content
    assert "def build_system(cfg, model, approver)" in develop
    assert ("Files", "read 1 file(s) you named: app.py") in seen


def test_missing_named_file_is_reported_not_guessed(tmp_path):
    reader = FileReferenceReader(base_dirs=(tmp_path,))
    model = standard_model("Medium|yes")
    build_app(Config(), model, file_reader=reader).orchestrator.handle(r"Read D:\nope\missing.py and explain it")
    respond = [c for c in model.calls if c.messages[0].content.startswith("Stage: respond.")][0].messages[-1].content
    assert "File not found" in respond and "do not guess" in respond


def test_files_to_create_are_not_reported_missing(tmp_path):
    reader = FileReferenceReader(base_dirs=(tmp_path,))
    assert reader.read_mentioned("Build site_tracker/tracker.html with a table") == []
    assert reader.read_mentioned(r"Build D:\out\new_page.html for me") == []


def test_reads_carry_over_exactly_one_turn(tmp_path):
    (tmp_path / "orchestrator.py").write_text("def handle(text):\n    return Reply(text)\n")
    reader = FileReferenceReader(base_dirs=(tmp_path,))
    model = standard_model("Medium|yes")
    orch = build_app(Config(), model, file_reader=reader).orchestrator
    orch.handle(f"Read {tmp_path / 'orchestrator.py'} please")
    n = len(model.calls)
    orch.handle("Using what you learned, write a server")
    second = [c for c in model.calls[n:] if c.messages[0].content.startswith("Stage: develop.")][0].messages[-1].content
    assert "def handle(text)" in second and "read in your previous turn" in second
    m = len(model.calls)
    orch.handle("Now something unrelated")
    third = [c for c in model.calls[m:] if c.messages[0].content.startswith("Stage: develop.")][0].messages[-1].content
    assert "def handle(text)" not in third                           # only one turn of carry-over


def test_path_detection():
    r = FileReferenceReader()
    assert r.candidates(r"First read D:\tessera\tessera\app.py and D:\tessera\tessera\__main__.py, then") == [
        r"D:\tessera\tessera\app.py", r"D:\tessera\tessera\__main__.py"]
    assert r.candidates('open "C:/My Docs/notes.txt" please') == ["C:/My Docs/notes.txt"]
    assert r.candidates("open tessera/app.py.") == ["tessera/app.py"]
    assert r.candidates("pi is 3.14, e.g. roughly") == []
