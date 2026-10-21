"""Patch 8 — from the real web_server.py run:
  * NameError on launch: section 1's run_server() call ended up above tessera_loader
  * code caught the failed import and silently "mocked" Tessera
  * Develop rewrote a saved file 4 times; Evaluate judged the draft text, not the file,
    and the final reply was a critique with no file path
"""

from pathlib import Path

from tessera.app import build_app
from tessera.config import Config
from tessera.context.tracker import ContextTracker
from tessera.core.approval import StaticApprover
from tessera.core.types import Observation
from tessera.llm.scripted import ScriptedModel
from tessera.tools.file_builder import CreateFileTool, SectionedFileBuilder
from tessera.tools.files import WriteFileTool
from tessera.tools.html_builder import HtmlPageBuilder
from tessera.tools.py_checks import import_problems, mock_problems, reorder_entrypoints

SETUP = '''import threading

LOCK = threading.Lock()


def tessera_loader():
    return "early copy"


run_server()
'''
LOADER = '''def tessera_loader():
    return {"ready": True}
'''
SERVER = '''def run_server():
    system = tessera_loader()
    return system
'''


class Gemma(ScriptedModel):
    def __init__(self, sections):
        super().__init__(context_window=32768)
        self.sections = sections
        self.when_agent("fileplan", "SECTION: setup | imports\nSECTION: tessera_loader | load\n"
                                    "SECTION: run_server | server\nNAMES: tessera_loader, run_server")
        self.when_agent("filesection", self._section)

    def _section(self, messages):
        body = messages[-1].content
        for i, text in enumerate(self.sections, 1):
            if f"section {i} —" in body:
                return text if isinstance(text, str) else text.pop(0)
        return "\n"


def tool(tmp_path, model):
    tracker = ContextTracker(model, 0.9)
    return CreateFileTool(SectionedFileBuilder(model, tracker, import_paths=(tmp_path,)),
                          WriteFileTool(tmp_path / "out", StaticApprover(True)), HtmlPageBuilder(model, tracker))


# --------------------------------------------------------- the real NameError
def test_startup_call_moved_below_definitions_and_file_imports(tmp_path):
    model = Gemma([SETUP, LOADER, SERVER])
    obs = tool(tmp_path, model).invoke({"path": "tessera_web/web_server.py", "description": "x"}, task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    src = (tmp_path / "out" / "tessera_web" / "web_server.py").read_text()
    assert src.rstrip().endswith("run_server()")
    assert src.index("def run_server") < src.rindex("run_server()")
    assert any("moved 1 start-up statement" in n for n in obs.notes)
    ns = {"__name__": "x"}
    exec(compile(src, "web_server.py", "exec"), ns)          # the exact crash from real use is gone
    assert ns["run_server"]() == {"ready": True}


def test_import_check_catches_crash_at_start_and_sends_it_back(tmp_path):
    broken = 'def run_server():\n    return tessera_loadr()\n\n\nVALUE = run_server()\n'   # typo + call at import
    fixed = 'def run_server():\n    return tessera_loader()\n'
    model = Gemma([SETUP.replace("run_server()\n", ""), LOADER, [broken, fixed]])
    obs = tool(tmp_path, model).invoke({"path": "w/app.py", "description": "x"}, task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    fix_prompts = [c.messages[-1].content for c in model.calls if "HAD THESE BUGS" in c.messages[-1].content]
    assert fix_prompts and "NameError" in fix_prompts[0] and "tessera_loadr" in fix_prompts[0]


def test_import_check_unit(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import json\n\ndef main():\n    return undefined_name\n\nif __name__ == '__main__':\n    main()\n")
    assert import_problems(f) == ([], [])                   # __main__ block not run → fine
    f.write_text("x = 1\nprint(y)\n")
    blocking, _ = import_problems(f)
    assert blocking and "NameError" in blocking[0] and "line 2" in blocking[0]
    f.write_text("import totally_missing_pkg_xyz\n")
    blocking, warnings = import_problems(f)
    assert blocking == [] and "totally_missing_pkg_xyz" in warnings[0]
    f.write_text("import time\nwhile True:\n    time.sleep(1)\n")
    blocking, _ = import_problems(f, timeout_s=2)
    assert "still running" in blocking[0]


def test_silent_mock_is_a_bug():
    src = ("try:\n    from tessera.core.approval import StaticApprover\nexcept ImportError:\n"
           "    print('Mocking')\n    class StaticApprover:\n        pass\n")
    assert mock_problems(src) and "mock" in mock_problems(src)[0]
    ok = "try:\n    import yaml\nexcept ImportError:\n    raise SystemExit('pip install pyyaml')\n"
    assert mock_problems(ok) == []


def test_reorder_leaves_library_calls_alone():
    texts, notes = reorder_entrypoints(["import logging\nlogging.basicConfig()\n", "def main():\n    pass\n"])
    assert texts[0] == "import logging\nlogging.basicConfig()\n" and notes == []


# ---------------------------------------------------- done after a saved file
class FileBridge:
    def __init__(self, tmp_path):
        self.tmp, self.calls = tmp_path, []

    def describe_capabilities(self):
        return "write_python_module(path, description)"

    def execute(self, req):
        self.calls.append(req.skill)
        p = self.tmp / Path(req.arguments["path"]).name
        p.write_text("def handle():\n    return 'ok'\n")
        return Observation("success", req.skill, result={"path": str(p)}, artifacts=(str(p),), notes=(f"saved to {p}",))


def test_saved_file_is_not_rewritten_and_evaluate_sees_it(tmp_path):
    writes = iter(['ACTION: write_python_module {"path": "tessera_web/web_server.py", "description": "x"}'] * 3
                  + ["Saved tessera_web/web_server.py."])
    model = (ScriptedModel().when_agent("router", "Complex|yes").when_agent("interpret", "i")
             .when_agent("develop", lambda m: next(writes)).when_agent("evaluate", "VERDICT: PASS")
             .when_agent("respond", "Saved."))
    bridge = FileBridge(tmp_path)
    build_app(Config(), model, bridge=bridge).orchestrator.handle("build web_server.py")
    assert bridge.calls == ["write_python_module"]                  # written once, never overwritten
    evaluate = [c for c in model.calls if c.messages[0].content.startswith("Stage: evaluate.")][0].messages[-1].content
    assert "SAVED FILES (the actual deliverable)" in evaluate and "def handle()" in evaluate
    respond = [c for c in model.calls if c.messages[0].content.startswith("Stage: respond.")][0].messages[-1].content
    assert str(tmp_path / "web_server.py") in respond


# ------------------------------------------------------ progress-aware budget
class Flaky:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def describe_capabilities(self):
        return "tool(x)"

    def execute(self, req):
        self.calls += 1
        return Observation("error", req.skill, stderr=self.errors.pop(0) if self.errors else "same error")


def _run(errors, n_actions):
    acts = iter([f'ACTION: tool {{"x": {i}}}' for i in range(n_actions)] + ["give up"] * 50)
    model = (ScriptedModel().when_agent("router", "Complex|yes").when_agent("interpret", "i")
             .when_agent("develop", lambda m: next(acts)).when_agent("evaluate", "VERDICT: PASS")
             .when_agent("respond", "r"))
    bridge = Flaky(errors)
    seen = []
    app = build_app(Config(), model, bridge=bridge)
    app.events.subscribe(lambda ch, msg: seen.append(msg))
    app.orchestrator.handle("q")
    return bridge, seen


def test_stuck_on_the_same_error_stops_after_three():
    bridge, seen = _run(["boom"] * 30, 30)
    assert bridge.calls == 3 and any("without progress" in s for s in seen)


def test_renumbered_errors_are_the_same_kind():
    bridge, _ = _run([f"NameError at line {i}: 'x{i}' is not defined" for i in range(30)], 30)
    assert bridge.calls == 3                                        # reworded/renumbered ≠ progress (patch 10)


def test_different_error_kinds_get_more_attempts_but_one_tool_stops_at_four():
    kinds = ["NameError: x", "TypeError: bad arg", "SyntaxError: eof", "KeyError: k", "ValueError: v"]
    bridge, seen = _run(kinds, 30)
    assert bridge.calls == 4 and any("without progress" in s for s in seen)


def test_example_config_matches_new_defaults():
    import tomllib
    data = tomllib.loads(Path(__file__).parents[1].joinpath("tessera.example.toml").read_text())
    assert data["max_skill_calls"] == 20 and data["max_stalled_calls"] == 3 and data["max_agent_iterations"] == 24
