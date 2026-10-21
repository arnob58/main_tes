"""Patch 9: the request's named requirements must be in the file; fakes are rejected.

Fixture = the real server Gemma wrote: no Tessera import, a "simulated" handler, plain text.
"""

from pathlib import Path

from tessera.context.tracker import ContextTracker
from tessera.core.approval import StaticApprover
from tessera.llm.scripted import ScriptedModel
from tessera.tools.file_builder import CreateFileTool, SectionedFileBuilder
from tessera.tools.files import WriteFileTool
from tessera.tools.html_builder import HtmlPageBuilder
from tessera.tools.requirements import extract_requirements, requirement_problems

REQUEST = ('Read D:\\tessera\\tessera\\__main__.py and D:\\tessera\\tessera\\core\\orchestrator.py, then build '
           'tessera_web/web_server.py using only Python\'s standard library: it loads Tessera like __main__.py does '
           '(load_config, load_model, build_system) but with StaticApprover(False) from tessera.core.approval, serves '
           'chat.html from its own folder at "/", and answers POST "/chat" JSON {"message": ...} with {"reply": ...} '
           'from system.app.orchestrator.handle(message).text. Use a lock so one message is processed at a time.')
FAKE = (Path(__file__).parent / "fixtures" / "fake_web_server_from_gemma.py").read_text()

REAL = '''import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def tessera_loader():
    from tessera.config import load_config
    from tessera.__main__ import load_model
    from tessera.app import build_system
    from tessera.core.approval import StaticApprover
    cfg = load_config("tessera.toml")
    return build_system(cfg, load_model(cfg), StaticApprover(False))


LOCK = threading.Lock()


def make_handler(system):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/chat":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:
                reply = system.app.orchestrator.handle(body["message"]).text
            data = json.dumps({"reply": reply}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)
    return Handler
'''


def test_extraction_from_the_real_request():
    tokens = [r.token for r in extract_requirements(REQUEST)]
    assert {"/chat", "StaticApprover", "load_config", "load_model", "build_system",
            "system.app.orchestrator.handle"} <= set(tokens)
    assert not {"does", "tessera_web", "web_server", "__main__"} & set(tokens)   # prose and paths ignored


def test_real_fake_server_is_rejected_with_specifics():
    problems = requirement_problems(FAKE, REQUEST, ".py")
    text = " ".join(problems)
    for name in ("StaticApprover", "load_model", "build_system", "system.app.orchestrator.handle"):
        assert name in text
    assert "simulat" in text


def test_correct_server_passes():
    assert requirement_problems(REAL, REQUEST, ".py") == []


def test_ordinary_requests_add_no_requirements():
    for req in ["Build a simple weather page that shows the temperature",
                "Make a to-do app that saves tasks in the browser",
                "Write a python script that renames my photos by date"]:
        assert extract_requirements(req) == []


def test_python_names_not_forced_on_html_but_paths_are():
    reqs = extract_requirements('Build chat.html that POSTs to "/chat" and shows reply_text from the JSON')
    page_without_path = "<script>fetch('/talk')</script>"
    assert any("/chat" in p for p in requirement_problems(page_without_path, 'POST to "/chat" with reply_text', ".html"))
    assert not any("reply_text" in p for p in requirement_problems("<script>fetch('/chat')</script>",
                                                                  'POST to "/chat" with reply_text', ".html"))
    assert reqs


def test_create_file_rejects_fake_then_accepts_real(tmp_path):
    replies = [FAKE, REAL]
    model = ScriptedModel(context_window=32768)
    model.when_agent("fileplan", "SECTION: server | the whole server\nNAMES: tessera_loader")
    model.when_agent("filesection", lambda m: replies.pop(0) if len(replies) > 1 else replies[0])
    tracker = ContextTracker(model, 0.9)
    tool = CreateFileTool(SectionedFileBuilder(model, tracker, import_check=False),
                          WriteFileTool(tmp_path / "out", StaticApprover(True)), HtmlPageBuilder(model, tracker))
    obs = tool.invoke({"path": "tessera_web/web_server.py", "description": "web server"}, task_id="t", purpose=REQUEST)
    assert obs.status == "success", obs.summary()
    saved = (tmp_path / "out" / "tessera_web" / "web_server.py").read_text()
    assert "orchestrator.handle" in saved and "Simulates" not in saved
    fix = [c.messages[-1].content for c in model.calls if "HAD THESE BUGS" in c.messages[-1].content][0]
    assert "StaticApprover" in fix and "simulat" in fix


def test_write_file_refuses_fake(tmp_path):
    obs = WriteFileTool(tmp_path, StaticApprover(True)).invoke(
        {"path": "web_server.py", "content": FAKE}, task_id="t", purpose=REQUEST)
    assert obs.status == "error" and "Requirement check" in obs.stderr and not (tmp_path / "web_server.py").exists()
