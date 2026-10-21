"""Phase 6: workflows, external apps / AI-to-AI, self-modification, task checkpoints."""

import json
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tessera.app import build_app
from tessera.config import Config
from tessera.context.checkpoint import CheckpointStore
from tessera.core.approval import StaticApprover
from tessera.core.task_state import TaskStatus
from tessera.core.types import Observation
from tessera.core.workflow import Step, WorkflowRunner, parse_steps
from tessera.llm.scripted import ScriptedModel
from tessera.tools.external import (AIConversation, ChatWithAppTool, CliAdapter, FindChatAppTool, HttpChatAdapter,
                                    PythonFunctionAdapter, discover_chat_apps)
from tessera.tools.self_improve import ImproveSourceTool, PatchError, SourceModifier, apply_edits, parse_edits

from conftest import standard_model


class Bridge:
    def __init__(self):
        self.calls = []

    def describe_capabilities(self):
        return "submit_form(url): submits a form"

    def execute(self, req):
        self.calls.append(req.skill)
        return Observation("success", req.skill, result="submitted, confirmation #123")


def workflow_rig(tmp_path, develop, approver):
    model = standard_model("Complex|yes")
    model.when_agent("develop", develop)
    model.rules.insert(0, model.rules.pop())
    bridge = Bridge()
    app = build_app(Config(), model, bridge=bridge)
    runner = WorkflowRunner(model, app.orchestrator._pipeline, CheckpointStore(tmp_path / "ck"), approver)
    return runner, model, bridge


STEPS = [Step(1, "Find the resume"), Step(2, "Read the job posting"), Step(3, "Submit the application", consequential=True)]


# ----------------------------------------------------------------- workflows
def test_parse_steps():
    steps = parse_steps('Plan: {"steps": [{"goal": "a"}, {"goal": "b", "consequential": true}, {"goal": ""}]}')
    assert [(s.id, s.goal, s.consequential) for s in steps] == [(1, "a", False), (2, "b", True)]


def test_workflow_pauses_for_confirmation_then_resumes_without_redoing(tmp_path):
    def develop(msgs):
        body = msgs[-1].content
        return 'ACTION: submit_form {"url": "x"}' if "Submit the application" in body.split("\n")[1] and "OBSERVATIONS" not in body else "done that"

    runner, model, bridge = workflow_rig(tmp_path, develop, StaticApprover(False))
    wf = runner.create("Apply to one job", STEPS)
    wf = runner.run(wf)
    assert wf.status == TaskStatus.WAITING_FOR_USER.value
    assert [s.status for s in wf.steps] == ["done", "done", "waiting_for_user"]
    assert bridge.calls == []                                  # nothing submitted
    calls_before = len(model.calls)

    reloaded = runner.load(wf.id)                              # e.g. after a restart
    assert reloaded.steps[0].status == "done"
    runner.approve(reloaded, 3)
    done = runner.run(reloaded)
    assert done.status == TaskStatus.COMPLETED.value and bridge.calls == ["submit_form"]
    assert done.steps[2].status == "done" and "confirmation #123" in done.steps[2].evidence[0]
    # steps 1-2 were not re-run: only step 3's passes were added
    new_stages = [c.messages[0].content.split("\n")[0] for c in model.calls[calls_before:]]
    assert new_stages.count("Stage: interpret.") == 1


def test_consequential_step_without_evidence_is_unverified(tmp_path):
    runner, _, _ = workflow_rig(tmp_path, lambda m: "I submitted it!", StaticApprover(True))
    wf = runner.run(runner.create("x", [Step(1, "Submit the form", consequential=True)]))
    assert wf.steps[0].status == "unverified"
    assert wf.status == TaskStatus.WAITING_FOR_USER.value     # not COMPLETED


def test_failed_step_is_checkpointed_and_listed(tmp_path):
    def develop(m):
        raise RuntimeError("GPU fell over")

    runner, _, _ = workflow_rig(tmp_path, develop, StaticApprover(True))
    wf = runner.run(runner.create("x", [Step(1, "a"), Step(2, "b")]))
    assert wf.status == TaskStatus.FAILED.value and "GPU fell over" in wf.steps[0].result
    assert [w.id for w in runner.pending()] == [wf.id]


def test_interrupted_task_checkpoint(tmp_path):
    class Boom(ScriptedModel):
        def generate(self, messages, **kw):
            if messages[0].content.startswith("Stage: respond."):
                raise RuntimeError("power cut")
            return super().generate(messages, **kw)

    model = Boom().when_agent("router", "Instant|no")
    store = CheckpointStore(tmp_path)
    app = build_app(Config(), model)
    app.orchestrator._checkpoints = store
    reply = app.orchestrator.handle("hello")
    saved = store.list("task_")
    assert reply.status is TaskStatus.FAILED and len(saved) == 1 and "power cut" in saved[0]["error"]
    ok = build_app(Config(), standard_model())
    ok.orchestrator._checkpoints = store
    ok.orchestrator.handle("hello")
    assert len(store.list("task_")) == 1                        # completed task left nothing behind


# ------------------------------------------------------ external apps / AI
LUNA = textwrap.dedent('''
    """GPT Luna — a tiny local chatbot."""
    def chat(message, history):
        turn = len(history) // 2 + 1
        return f"Luna (turn {turn}) heard: {message}"

    if __name__ == "__main__":
        while True:
            print(chat(input("> "), []))
''')


@pytest.fixture
def desktop(tmp_path):
    d = tmp_path / "Desktop"
    (d / "GPT Luna").mkdir(parents=True)
    (d / "GPT Luna" / "luna_bot.py").write_text(LUNA)
    (d / "notes").mkdir()
    (d / "notes" / "todo.txt").write_text("x")
    echo = d / "echo_bot.py"
    echo.write_text("import sys\nline = sys.stdin.readline().strip()\nwhile True:\n    print('ECHO:', line)\n    break\n")
    return d


def test_discovery_finds_interface_by_ast(desktop):
    cands = discover_chat_apps(desktop, "GPT Luna")
    best = cands[0]
    assert best.kind == "python_function" and best.detail["function"] == "chat" and best.detail["takes_history"]
    assert discover_chat_apps(desktop, "nonexistent") == []


def test_python_function_adapter_runs_out_of_process(desktop):
    a = PythonFunctionAdapter(str(desktop / "GPT Luna" / "luna_bot.py"), "chat", takes_history=True)
    assert a.send("hi", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]) == "Luna (turn 2) heard: hi"


def test_cli_adapter(desktop):
    assert CliAdapter([sys.executable, str(desktop / "echo_bot.py")]).send("ping", []) == "ECHO: ping"


def test_ai_conversation_keeps_state_and_terminates(desktop):
    mine = iter(["Hello Luna, what can you do?", "Tell me one fact.", "[END]"])
    model = ScriptedModel().when_agent("converse", lambda m: next(mine))
    adapter = PythonFunctionAdapter(str(desktop / "GPT Luna" / "luna_bot.py"), "chat", takes_history=True)
    res = AIConversation(model, adapter, name="GPT Luna", max_turns=6).run("get to know Luna")
    assert res.status == "finished"
    assert [t.speaker for t in res.turns] == ["tessera", "GPT Luna", "tessera", "GPT Luna"]
    assert res.turns[3].text.startswith("Luna (turn 2)")            # session history reached the app
    last_prompt = model.calls[-1].messages
    assert [m.role.value for m in last_prompt[1:]] == ["assistant", "user", "assistant", "user"]


def test_conversation_turn_limit(desktop):
    model = ScriptedModel().when_agent("converse", "keep going")
    adapter = CliAdapter([sys.executable, str(desktop / "echo_bot.py")])
    res = AIConversation(model, adapter, name="echo", max_turns=3).run("chat")
    assert res.status == "turn_limit" and len(res.turns) == 6


def test_http_adapter_and_tools(desktop):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"data": {"reply": body["text"].upper()}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        a = HttpChatAdapter(f"http://127.0.0.1:{srv.server_port}/chat", message_field="text", reply_path="data.reply")
        assert a.send("hi", []) == "HI"
    finally:
        srv.shutdown()

    found = FindChatAppTool().invoke({"name": "GPT Luna", "root": str(desktop)}, task_id="t", purpose="")
    cand = found.result[0]
    model = ScriptedModel().when_agent("converse", "[END]")
    denied = ChatWithAppTool(model, StaticApprover(False)).invoke(
        {"kind": cand["kind"], "detail": cand["detail"], "name": "GPT Luna"}, task_id="t", purpose="chat")
    assert denied.status == "denied"
    ok = ChatWithAppTool(model, StaticApprover(True)).invoke(
        {"kind": cand["kind"], "detail": cand["detail"], "name": "GPT Luna", "opening": "Hi!", "max_turns": 2},
        task_id="t", purpose="chat")
    assert ok.status == "success" and "Luna (turn 1) heard: Hi!" in ok.stdout


# ------------------------------------------------------------ self-modification
@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mathx.py").write_text("def mean(xs):\n    return sum(xs) / len(xs)\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_mathx.py").write_text(
        "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parents[1]))\n"
        "from pkg.mathx import mean\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n\n"
        "def test_empty():\n    assert mean([]) == 0\n")
    return root


GOOD = """<<<<<<< SEARCH
def mean(xs):
    return sum(xs) / len(xs)
=======
def mean(xs):
    if not xs:
        return 0
    return sum(xs) / len(xs)
>>>>>>> REPLACE
WHY: mean([]) raised ZeroDivisionError"""

BAD = """<<<<<<< SEARCH
    return sum(xs) / len(xs)
=======
    return sum(xs)
>>>>>>> REPLACE
WHY: simpler"""


def test_self_patch_tested_and_kept_but_not_active(project, tmp_path):
    original_test = (project / "tests" / "test_mathx.py").read_text()
    model = ScriptedModel().when_agent("selfpatch", GOOD)
    mod = SourceModifier(project, tmp_path / "backups", StaticApprover(True))
    obs = ImproveSourceTool(model, mod).invoke({"path": "pkg/mathx.py", "goal": "handle empty lists"}, task_id="t", purpose="")
    assert obs.status == "success", obs.stderr
    assert "if not xs" in (project / "pkg" / "mathx.py").read_text()
    assert "restart Tessera to activate" in obs.result and "tests=passed" in obs.result
    assert Path(obs.artifacts[0]).read_text() == "def mean(xs):\n    return sum(xs) / len(xs)\n"   # backup
    assert "+    if not xs:" in obs.stdout
    assert (project / "tests" / "test_mathx.py").read_text() == original_test


def test_failing_patch_is_rolled_back(project, tmp_path):
    before = (project / "pkg" / "mathx.py").read_text()
    model = ScriptedModel().when_agent("selfpatch", BAD)
    obs = ImproveSourceTool(model, SourceModifier(project, tmp_path / "b", StaticApprover(True))).invoke(
        {"path": "pkg/mathx.py", "goal": "x"}, task_id="t", purpose="")
    assert obs.status == "error" and "rolled back" in obs.result and "tests=FAILED" in obs.result
    assert (project / "pkg" / "mathx.py").read_text() == before


def test_patch_guards(project, tmp_path):
    mod = SourceModifier(project, tmp_path / "b", StaticApprover(False))
    edits, _ = parse_edits(GOOD)
    rep = mod.apply("pkg/mathx.py", edits)
    assert not rep.applied and "if not xs" not in (project / "pkg" / "mathx.py").read_text()
    with pytest.raises(PatchError, match="found 2 times"):
        apply_edits("a\na\n", parse_edits("<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE")[0])
    with pytest.raises(PatchError, match="outside"):
        mod.apply("../../etc/passwd.py", edits)


def test_hot_reloadable_dir_reports_active_and_rollback(project, tmp_path):
    mod = SourceModifier(project, tmp_path / "b", StaticApprover(True), hot_dirs=(project / "pkg",))
    rep = mod.apply("pkg/mathx.py", parse_edits(GOOD)[0])
    assert rep.active_now and not rep.rolled_back
    mod.rollback(rep.backup)
    assert "if not xs" not in (project / "pkg" / "mathx.py").read_text()
