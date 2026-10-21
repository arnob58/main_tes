"""Phase 5: server backend, vision adapters, GUI loop.

No display or vision model exists in CI, so the GUI driver and the vision
model are fakes; the loop logic, safety gates and coordinate handling are real.
The llama-server client is exercised against a local HTTP server that mimics
the documented endpoints. Two tests use the *installed* llama-cpp-python.
"""

import base64
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tessera.config import Config
from tessera.core.approval import StaticApprover
from tessera.core.types import Message
from tessera.execution.artifacts import WorkspaceManager
from tessera.execution.desktop_loop import DesktopLoop, parse_step
from tessera.execution.gui_adapter import ScreenSize
from tessera.llm.server_backend import LlamaServerEngine
from tessera.llm.vision import ModelVision, UnsupportedVision, VisionObservation
from tessera.tools.desktop import DesktopTaskTool, LookAtScreenTool


def tiny_png(path: Path, w=4, h=3):
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class FakeDriver:
    def __init__(self, size=ScreenSize(1920, 1080), ok=True):
        self.size, self.ok, self.actions = size, ok, []
        self.resize_after_shot = None

    def available(self):
        return (True, "ok") if self.ok else (False, "no display")

    def screen_size(self):
        return self.size

    def screenshot(self, path):
        tiny_png(path)
        shot = self.size
        if self.resize_after_shot:
            self.size, self.resize_after_shot = self.resize_after_shot, None
        return shot

    def click(self, px, py, *, button="left", clicks=1):
        self.actions.append(("click", px, py, clicks))

    def type_text(self, text):
        self.actions.append(("type", text))

    def press(self, key):
        self.actions.append(("press", key))

    def hotkey(self, *keys):
        self.actions.append(("hotkey", keys))

    def scroll(self, amount):
        self.actions.append(("scroll", amount))

    def open_url(self, url, browser=""):
        self.actions.append(("open_url", url, browser))

    def launch(self, app):
        self.actions.append(("launch", app))


class FakeVision:
    supported = True

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def observe(self, image_path, question, *, max_output_tokens=300):
        assert Path(image_path).read_bytes().startswith(b"\x89PNG")
        self.seen.append((str(image_path), question))
        return VisionObservation("ok", self.replies.pop(0), str(image_path))


def step(obs, **action):
    return json.dumps({"observation": obs, "action": action})


JOB_FLOW = [
    step("Job site home page with a search box", type="click", x=500, y=100, target="search box"),
    step("Search box focused", type="type", text="Social Media Coordinator Toronto"),
    step("Query typed", type="press", key="enter"),
    step("Results list; first job has an Apply button", type="click", x=800, y=420, target="Easy Apply - Submit application", consequential=True),
    step("Application submitted confirmation visible", type="done", summary="Application submitted (confirmation shown)"),
]


def loop(driver, vision, approver):
    return DesktopLoop(driver, vision, approver, settle_s=0)


def test_observe_act_loop_with_approvals(tmp_path):
    driver, vision, approver = FakeDriver(), FakeVision(JOB_FLOW), StaticApprover(True)
    res = loop(driver, vision, approver).run("Apply to a Social Media Coordinator job", tmp_path,
                                             start_url="https://jobs.example", browser="chrome")
    assert res.status == "done" and "confirmation" in res.summary
    assert driver.actions[0] == ("open_url", "https://jobs.example", "chrome")
    assert driver.actions[1] == ("click", round(0.5 * 1919), round(0.1 * 1079), 1)
    assert ("type", "Social Media Coordinator Toronto") in driver.actions
    assert [a for a, _ in approver.asked] == ["gui_type", "gui_consequential"]
    # each action is tied to the screenshot it was chosen from
    assert [Path(s.screenshot).name for s in res.steps] == [f"step_00{i}.png" for i in range(1, 6)]
    assert "1. " in vision.seen[1][1] and "clicked search box" in vision.seen[1][1]


def test_consequential_click_denied_stops_without_clicking(tmp_path):
    driver = FakeDriver()
    approver_typing = type("A", (), {"asked": [], "approve": lambda s, a, d: s.asked.append(a) or a == "gui_type"})()
    res = loop(driver, FakeVision(JOB_FLOW), approver_typing).run("apply", tmp_path)
    assert res.status == "denied"
    clicks = [a for a in driver.actions if a[0] == "click"]
    assert len(clicks) == 1  # only the search box; the submit click never happened


def test_resolution_independent_coordinates():
    assert ScreenSize(1920, 1080).to_pixels(500, 500) == (960, 540)
    assert ScreenSize(1366, 768).to_pixels(500, 500) == (682, 384)
    assert ScreenSize(1366, 768).to_pixels(1000, 1000) == (1365, 767)
    with pytest.raises(ValueError):
        ScreenSize(100, 100).to_pixels(1200, 5)


def test_stale_screenshot_is_not_clicked(tmp_path):
    driver = FakeDriver()
    driver.resize_after_shot = ScreenSize(1280, 720)
    replies = [step("x", type="click", x=10, y=10, target="menu"), step("y", type="done", summary="ok")]
    res = loop(driver, FakeVision(replies), StaticApprover(True)).run("g", tmp_path)
    assert res.steps[0].outcome.startswith("skipped: screen size changed")
    assert not [a for a in driver.actions if a[0] == "click"]


def test_unsupported_vision_and_display_are_explicit(tmp_path):
    res = loop(FakeDriver(), UnsupportedVision("text-only model"), StaticApprover()).run("g", tmp_path)
    assert res.status == "unsupported" and "text-only model" in res.summary
    res = loop(FakeDriver(ok=False), FakeVision([]), StaticApprover()).run("g", tmp_path)
    assert res.status == "unsupported" and "no display" in res.summary


def test_unparseable_replies_fail_after_two(tmp_path):
    res = loop(FakeDriver(), FakeVision(["I see a page", "still prose"]), StaticApprover()).run("g", tmp_path)
    assert res.status == "failed" and len(res.steps) == 2


def test_macro_runs_without_model_calls(tmp_path):
    vision = FakeVision([step("done", type="done", summary="ok")])
    driver = FakeDriver()
    res = loop(driver, vision, StaticApprover(True)).run(
        "g", tmp_path, macro=[{"type": "hotkey", "keys": ["ctrl", "l"]}, {"type": "type", "text": "example.com"}, {"type": "press", "key": "enter"}])
    assert driver.actions[:3] == [("hotkey", ("ctrl", "l")), ("type", "example.com"), ("press", "enter")]
    assert len(vision.seen) == 1 and res.status == "done"


def test_step_limit_never_claims_success(tmp_path):
    replies = [step("x", type="scroll", amount=-3)] * 3
    res = DesktopLoop(FakeDriver(), FakeVision(replies), StaticApprover(), settle_s=0, max_steps=3).run("g", tmp_path)
    assert res.status == "step_limit"


def test_parse_step():
    assert parse_step('Sure: {"observation": "a", "action": {"type": "wait"}}') == ("a", {"type": "wait"})
    with pytest.raises(ValueError):
        parse_step('{"observation": "a"}')


def test_desktop_tools_map_status(tmp_path):
    ws = WorkspaceManager(tmp_path)
    tool = DesktopTaskTool(loop(FakeDriver(), FakeVision(JOB_FLOW), StaticApprover(True)), ws)
    obs = tool.invoke({"goal": "apply"}, task_id="t", purpose="apply")
    assert obs.status == "success" and obs.result["status"] == "done" and obs.artifacts
    look = LookAtScreenTool(FakeDriver(), UnsupportedVision("no mmproj"), ws)
    assert look.invoke({}, task_id="t", purpose="").status == "unavailable"


# ------------------------------------------------------- llama-server client
class FakeServer(BaseHTTPRequestHandler):
    log = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeServer.log.append(("GET", self.path, None))
        if self.path == "/props":
            self._send({"default_generation_settings": {"n_ctx": 2048}})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "gemma", "capabilities": ["completion", "multimodal"]}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        FakeServer.log.append(("POST", self.path, body))
        if self.path == "/apply-template":
            self._send({"prompt": "<bos>" + "|".join(m["content"] for m in body["messages"]) + "<model>"})
        elif self.path == "/tokenize":
            self._send({"tokens": list(range(len(body["content"].split("|")) + 2))})
        elif self.path == "/completion":
            limit = body["n_predict"] < 3
            self._send({"content": " Instant|no ", "tokens_evaluated": len(body["prompt"]),
                        "tokens_predicted": 2, "stop_type": "limit" if limit else "eos"})
        elif self.path.startswith("/slots/0?action=erase"):
            self._send({"id_slot": 0, "n_erased": 10})
        elif self.path == "/v1/chat/completions":
            self._send({"choices": [{"message": {"content": "A login form."}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 300, "completion_tokens": 4}})


@pytest.fixture
def server():
    FakeServer.log = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_server_engine_exact_tokens_and_grammar(server):
    eng = LlamaServerEngine(server)
    assert eng.context_window == 2048
    msgs = [Message.system("s"), Message.user("u")]
    assert eng.count_prompt_tokens(msgs) == 4
    r = eng.generate(msgs, max_output_tokens=8, grammar='root ::= "x"')
    sent = [b for m, p, b in FakeServer.log if p == "/completion"][0]
    assert sent["prompt"] == [0, 1, 2, 3] and sent["grammar"] == 'root ::= "x"' and sent["n_predict"] == 8
    tok_req = [b for m, p, b in FakeServer.log if p == "/tokenize"][0]
    assert tok_req["parse_special"] is True and tok_req["add_special"] is True
    assert (r.text, r.input_tokens, r.output_tokens, r.finish_reason) == ("Instant|no", 4, 2, "stop")
    assert eng.generate(msgs, max_output_tokens=2).finish_reason == "length"
    eng.reset_context()
    assert FakeServer.log[-1][1] == "/slots/0?action=erase"


def test_server_vision_sends_image_data(server, tmp_path):
    img = tmp_path / "s.png"
    tiny_png(img)
    vision = ModelVision(LlamaServerEngine(server))
    assert vision.supported
    seen = vision.observe(img, "what is this?")
    assert seen.status == "ok" and seen.text == "A login form." and seen.input_tokens == 300
    body = [b for m, p, b in FakeServer.log if p == "/v1/chat/completions"][0]
    part = body["messages"][0]["content"][0]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) == img.read_bytes()


def test_unreachable_server_is_a_clear_error():
    from tessera.llm.server_backend import ServerError
    with pytest.raises(ServerError, match="cannot reach"):
        LlamaServerEngine("http://127.0.0.1:9", timeout_s=2)


# ---------------------------------------------------- real llama-cpp-python
llama_cpp = pytest.importorskip("llama_cpp")


def test_real_library_has_gemma4_vision_handler(tmp_path):
    from tessera.llm.engine import vision_handler_for
    assert vision_handler_for(Config())[1] == "no mmproj_path configured"
    assert "not found" in vision_handler_for(Config(mmproj_path=str(tmp_path / "x.gguf")))[1]
    from llama_cpp.llama_chat_format import Gemma4ChatHandler, MTMDChatHandler
    assert issubclass(Gemma4ChatHandler, MTMDChatHandler)


GEMMA_LIKE_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
    "{{ m['content'] }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def test_engine_with_real_formatter_and_grammar():
    """Fake weights, real llama-cpp-python formatter + GBNF compiler."""
    from tessera.llm.engine import LlamaCppEngine
    from tessera.llm.prompts import ROUTER_GRAMMAR

    class Weights:
        metadata = {"tokenizer.chat_template": GEMMA_LIKE_TEMPLATE}
        chat_handler = None
        def token_bos(self): return 2
        def token_eos(self): return 1
        def detokenize(self, ids, special=False): return {2: b"<bos>", 1: b"<eos>"}[ids[0]]
        def tokenize(self, data, add_bos=True, special=False):
            self.rendered = data.decode()
            return [2] + [9] * len(data.split()) if self.rendered.startswith("<bos>") else [9] * len(data.split())
        def create_completion(self, **kw):
            self.kw = kw
            return {"choices": [{"text": "Medium|yes", "finish_reason": "stop"}], "usage": {"prompt_tokens": len(kw["prompt"]), "completion_tokens": 3}}
        def close(self): pass

    w = Weights()
    eng = LlamaCppEngine(Config(), llama_factory=lambda c: w)
    try:
        r = eng.generate([Message.system("Classify."), Message.user("hi")], grammar=ROUTER_GRAMMAR)
        assert w.rendered.startswith("<bos><start_of_turn>system\nClassify.<end_of_turn>")
        assert w.rendered.endswith("<start_of_turn>model\n")
        assert type(w.kw["grammar"]).__name__ == "LlamaGrammar"
        assert "<eos>" in w.kw["stop"] and r.text == "Medium|yes"
        assert w.kw["prompt"].count(2) == 1  # exactly one BOS
        ok, why = eng.vision_status()
        assert not ok and why == "no mmproj_path configured"
    finally:
        eng.close()
