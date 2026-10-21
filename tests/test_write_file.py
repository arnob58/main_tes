"""The reported case: "Build a project: a single HTML page called tracker.html …" must produce a real file."""

from pathlib import Path

from tessera.agents.parsing import parse_action
from tessera.app import build_system
from tessera.config import Config
from tessera.core.approval import StaticApprover
from tessera.llm.scripted import ScriptedModel
from tessera.memory.embedding_service import HashingEmbedder
from tessera.tools.files import WriteFileTool

PROMPT = ("Build a project: a single HTML page called tracker.html that lets me add websites with a name, URL and "
          "notes, shows them in a table, and saves them in the browser. Put it in a folder called site_tracker.")

HTML = """<!DOCTYPE html>
<html><head><title>Site Tracker</title></head>
<body>
<form id="f"><input id="name"><input id="url"><textarea id="notes"></textarea><button>Add</button></form>
<table id="t"><thead><tr><th>Name</th><th>URL</th><th>Notes</th></tr></thead><tbody></tbody></table>
<script>
const KEY = "sites";
const load = () => JSON.parse(localStorage.getItem(KEY) || "[]");
const save = s => localStorage.setItem(KEY, JSON.stringify(s));
</script>
</body></html>"""

DEVELOP_FIRST = f'The page is ready.\nACTION: write_file {{"path": "site_tracker/tracker.html"}}\n```html\n{HTML}\n```\n'


def test_parser_takes_code_block_as_content():
    a = parse_action(DEVELOP_FIRST)
    assert a.request.skill == "write_file" and a.request.arguments["path"] == "site_tracker/tracker.html"
    assert a.request.arguments["content"] == HTML + "\n" and a.preamble == "The page is ready."
    assert parse_action('ACTION: calculator {"expression": "1+1"}').request.arguments == {"expression": "1+1"}


def test_unclosed_block_is_refused_not_half_saved():
    a = parse_action('ACTION: write_file {"path": "x.html"}\n```html\n<html><body>')
    assert a.request is None and "never closed" in a.error


def test_reported_prompt_writes_the_file(tmp_path):
    out = tmp_path / "Documents" / "Tessera"
    cfg = Config(database_path=str(tmp_path / "db.sqlite"), working_directory=str(tmp_path / "w"),
                 skill_directory=str(tmp_path / "g"), checkpoint_directory=str(tmp_path / "ck"),
                 output_directory=str(out), embedder="hashing", gui_enabled=False)
    develop = iter([DEVELOP_FIRST, "Saved site_tracker/tracker.html; it stores sites in localStorage."])
    model = (ScriptedModel()
             .when_agent("router", "Complex|yes")
             .when_agent("interpret", "INTENT: build an HTML site tracker\nTOOLS: write_file")
             .when_agent("develop", lambda m: next(develop))
             .when_agent("evaluate", "VERDICT: PASS")
             .when_agent("respond", "Created site_tracker/tracker.html in your Tessera folder."))
    system = build_system(cfg, model, StaticApprover(False), embedder=HashingEmbedder())
    system.app.orchestrator.handle(PROMPT)

    page = out / "site_tracker" / "tracker.html"
    assert page.exists(), "the file must exist on disk"
    text = page.read_text()
    assert "<table" in text and "localStorage" in text and text.startswith("<!DOCTYPE html>")
    develop_prompts = [c.messages for c in model.calls if c.messages[0].content.startswith("Stage: develop.")]
    assert "write_file(" in develop_prompts[0][-1].content                 # tool was offered
    assert "you MUST save it" in develop_prompts[0][0].content
    assert "status=success" in develop_prompts[1][-1].content and str(page) in develop_prompts[1][-1].content


def test_write_file_guards(tmp_path):
    asked = StaticApprover(False)
    tool = WriteFileTool(tmp_path / "out", asked)
    ok = tool.invoke({"path": "a/b.txt", "content": "hi"}, task_id="t", purpose="")
    assert ok.status == "success" and (tmp_path / "out" / "a" / "b.txt").read_text() == "hi\n"
    assert tool.invoke({"path": "a/b.txt", "content": "hi"}, task_id="t", purpose="").result["unchanged"]
    over = tool.invoke({"path": "a/b.txt", "content": "changed"}, task_id="t", purpose="")
    assert over.status == "denied" and (tmp_path / "out" / "a" / "b.txt").read_text() == "hi\n"
    outside = tool.invoke({"path": str(tmp_path / "elsewhere.txt"), "content": "x"}, task_id="t", purpose="")
    assert outside.status == "denied" and not (tmp_path / "elsewhere.txt").exists()
    assert {a for a, _ in asked.asked} == {"overwrite_file", "write_outside_output"}
    assert tool.invoke({"path": "c.txt"}, task_id="t", purpose="").status == "error"
    assert not Path(tmp_path / "out" / "c.txt").exists()
