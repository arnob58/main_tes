"""Patch 3: Tessera finds bugs in what it writes and keeps fixing instead of stopping.

The broken page fixture is the exact tracker.html Gemma produced in real use.
"""

import shutil
from pathlib import Path

import pytest

from tessera.agents.parsing import parse_action
from tessera.app import build_app, build_system
from tessera.config import Config
from tessera.core.approval import StaticApprover
from tessera.core.types import Observation
from tessera.llm.scripted import ScriptedModel
from tessera.memory.embedding_service import HashingEmbedder
from tessera.tools.validators import bracket_problems, validate_file

BROKEN = (Path(__file__).parent / "fixtures" / "broken_tracker_from_gemma.html").read_text()

GOOD = """<!DOCTYPE html>
<html><head><title>Site Tracker</title>
<style>body { font-family: sans-serif; } table { width: 100%; }</style></head>
<body>
<form id="f"><input id="name" required><input id="url" required><textarea id="notes"></textarea><button>Add</button></form>
<table><thead><tr><th>Name</th><th>URL</th><th>Notes</th></tr></thead><tbody id="rows"></tbody></table>
<script>
const KEY = "sites";
const load = () => JSON.parse(localStorage.getItem(KEY) || "[]");
function render() {
  const rows = document.getElementById("rows");
  rows.innerHTML = "";
  for (const s of load()) {
    const tr = rows.insertRow();
    [s.name, s.url, s.notes].forEach(v => { tr.insertCell().textContent = v; });
  }
}
document.getElementById("f").addEventListener("submit", e => {
  e.preventDefault();
  const sites = load();
  sites.push({ name: name.value, url: url.value, notes: notes.value });
  localStorage.setItem(KEY, JSON.stringify(sites));
  render();
});
render();
</script>
</body></html>
"""


def write_action(html: str) -> str:
    return f'ACTION: write_file {{"path": "site_tracker/tracker.html"}}\n```html\n{html}```\n'


# ------------------------------------------------------------------ validators
def test_validator_catches_every_bug_in_the_real_broken_page():
    problems = validate_file("tracker.html", BROKEN)
    text = "\n".join(problems)
    assert "stray markdown fence" in text
    assert "<script>" in text and "never closed" in text
    assert "CSS" in text
    assert "JavaScript" in text   # the duplicated block makes the script itself invalid


def test_validator_passes_a_correct_page():
    assert validate_file("tracker.html", GOOD) == []


@pytest.mark.parametrize("name, text, needle", [
    ("data.json", '{"a": 1,}', "JSON"),
    ("app.py", "def f(:\n", "Python"),
    ("site.css", "body { color: red;", "CSS"),
    ("app.js", "function f() { return 1;", "JavaScript"),
    ("notes.md", "```py\nx\n```", None),          # fences are fine in markdown
])
def test_validators_by_type(name, text, needle):
    problems = validate_file(name, text)
    assert (needle in " ".join(problems)) if needle else problems == []


def test_bracket_fallback_without_node():
    assert bracket_problems("let s = '}'; f(a, [1, 2]);", "JavaScript") == []
    assert "never closed" in bracket_problems("if (x) { y(); ", "JavaScript")[0]
    assert "unexpected" in bracket_problems("a); //", "JavaScript")[0]


# --------------------------------------------------------- fix-until-it-works
def system_with(tmp_path, develop_replies, approver=None):
    out = tmp_path / "out"
    cfg = Config(database_path=str(tmp_path / "db.sqlite"), working_directory=str(tmp_path / "w"),
                 skill_directory=str(tmp_path / "g"), checkpoint_directory=str(tmp_path / "ck"),
                 output_directory=str(out), embedder="hashing", gui_enabled=False,
                 agent_max_tokens=4000)  # scripted model counts words; keep whole pages in one reply
    replies = iter(develop_replies)
    model = (ScriptedModel(context_window=32768)
             .when_agent("router", "Complex|yes")
             .when_agent("interpret", "INTENT: build tracker.html")
             .when_agent("develop", lambda m: next(replies))
             .when_agent("evaluate", "VERDICT: PASS")
             .when_agent("respond", "Saved your tracker."))
    system = build_system(cfg, model, approver or StaticApprover(False), embedder=HashingEmbedder())
    return system, model, out / "site_tracker" / "tracker.html"


def develop_prompts(model):
    return [c.messages[-1].content for c in model.calls if c.messages[0].content.startswith("Stage: develop.")]


def test_buggy_file_is_rejected_then_fixed(tmp_path):
    system, model, page = system_with(tmp_path, [write_action(BROKEN), write_action(GOOD), "Saved site_tracker/tracker.html."])
    system.app.orchestrator.handle("Build tracker.html in site_tracker")
    assert page.read_text() == GOOD                         # only the corrected file was ever written
    second = develop_prompts(model)[1]
    assert "NOT SAVED — the file has bugs" in second and "JavaScript" in second


def test_code_printed_but_not_saved_gets_a_nudge(tmp_path):
    printed = "Here is the page:\n```html\n" + GOOD + "```"
    system, model, page = system_with(tmp_path, [printed, write_action(GOOD), "Saved."])
    system.app.orchestrator.handle("Build tracker.html in site_tracker")
    assert page.exists()
    assert "did not save it" in develop_prompts(model)[1]


def test_nudge_only_once_then_answer(tmp_path):
    printed = "```html\n" + GOOD + "```"
    system, model, page = system_with(tmp_path, [printed, printed])
    reply = system.app.orchestrator.handle("Build tracker.html in site_tracker")
    assert not page.exists() and reply.text == "Saved your tracker."
    assert len(develop_prompts(model)) == 2


def test_multiline_json_with_raw_newlines_is_accepted():
    text = 'ACTION: write_file {"path": "a/b.html",\n "content": "<p>\n hi\n</p>"}\nmore text'
    a = parse_action(text)
    assert a.request and a.request.arguments["content"] == "<p>\n hi\n</p>"


def test_bad_json_error_shows_the_correct_format():
    a = parse_action('ACTION: write_file {"path": broken')
    assert a.request is None and "fenced code block" in a.error and 'ACTION: write_file {"path"' in a.error


def test_exact_repeat_of_failed_call_is_blocked():
    class Bridge:
        def __init__(self):
            self.n = 0

        def describe_capabilities(self):
            return "flaky(x)"

        def execute(self, req):
            self.n += 1
            return Observation("error", req.skill, stderr="boom")

    replies = iter(['ACTION: flaky {"x": 1}', 'ACTION: flaky {"x": 1}', 'ACTION: flaky {"x": 2}', "gave up"])
    model = (ScriptedModel().when_agent("router", "Complex|yes").when_agent("interpret", "i")
             .when_agent("develop", lambda m: next(replies)).when_agent("evaluate", "VERDICT: PASS")
             .when_agent("respond", "r"))
    bridge = Bridge()
    build_app(Config(), model, bridge=bridge).orchestrator.handle("q")
    assert bridge.n == 2                                     # x=1 once, x=2 once; the exact repeat never ran
    assert "already failed" in develop_prompts(model)[2]


def test_develop_prompt_asks_for_new_skills_and_persistence():
    from tessera.llm.prompts import DEVELOP
    p = DEVELOP.system_prompt
    assert "request one anyway" in p and "do not give up after one failure" in p


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_node_check_reports_line():
    problems = validate_file("x.js", "const a = 1;\nconst a = 2;\n")
    assert problems and "JavaScript" in problems[0]
