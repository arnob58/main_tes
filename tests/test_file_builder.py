"""Patch 4: plain FILE blocks, lenient content, and create_file (section-by-section builds).

The scripted "model" plays Gemma: it writes each section from what the prompt shows,
including one buggy section and one cut-off section, to exercise fix and continuation.
"""

import re

from tessera.agents.parsing import parse_action
from tessera.app import build_system
from tessera.config import Config
from tessera.context.tracker import ContextTracker
from tessera.core.approval import StaticApprover
from tessera.llm.scripted import ScriptedModel
from tessera.memory.embedding_service import HashingEmbedder
from tessera.tools.file_builder import CreateFileTool, SectionedFileBuilder, parse_outline
from tessera.tools.files import WriteFileTool
from tessera.tools.validators import validate_file

HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Weather Tracker</title>
<style>
body { margin: 0; font-family: system-ui; background: linear-gradient(135deg, #a1c4fd, #c2e9fb); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
.card { background: rgba(255,255,255,.35); backdrop-filter: blur(10px); border-radius: 18px; padding: 16px; }
.card:hover { transform: translateY(-4px); }
</style></head>
"""
MARKUP = """<body>
<h1>Weather Tracker</h1>
<form id="cityForm"><input id="cityInput" placeholder="Add a city"><button>Add</button></form>
<p id="empty">No cities yet — add one above.</p>
<div id="cities" class="grid"></div>
"""
SCRIPT_BUGGY = """<script>
const KEY = "cities";
function load() { return JSON.parse(localStorage.getItem(KEY) || "[]");
</script>
</body></html>
"""
SCRIPT_GOOD = """<script>
const KEY = "cities";
const ICON = c => c === 0 ? "☀️" : c < 4 ? "⛅" : c < 70 ? "🌧️" : "❄️";
function load() { return JSON.parse(localStorage.getItem(KEY) || "[]"); }
function save(list) { localStorage.setItem(KEY, JSON.stringify(list)); }
async function weatherFor(city) {
  const g = await (await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`)).json();
  const { latitude, longitude, name } = g.results[0];
  const w = await (await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,weather_code,wind_speed_10m`)).json();
  return { name, ...w.current };
}
async function render() {
  const box = document.getElementById("cities");
  const list = load();
  document.getElementById("empty").hidden = list.length > 0;
  box.innerHTML = "";
  for (const city of list) {
    const d = await weatherFor(city);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h2>${ICON(d.weather_code)} ${d.name}</h2><p>${d.temperature_2m} °C · wind ${d.wind_speed_10m} km/h</p>`;
    const rm = document.createElement("button");
    rm.textContent = "Remove";
    rm.onclick = () => { save(load().filter(c => c !== city)); render(); };
    card.append(rm);
    box.append(card);
  }
}
document.getElementById("cityForm").addEventListener("submit", e => {
  e.preventDefault();
  const c = document.getElementById("cityInput").value.trim();
  if (c) { save([...load(), c]); render(); }
});
render();
</script>
</body></html>
"""
OUTLINE = """SECTION: head and css | doctype, head, glass-card CSS
SECTION: markup | body, form #cityForm, input #cityInput, #empty message, #cities grid
SECTION: script | localStorage key cities, Open-Meteo fetches, render(), remove buttons
NAMES: cityForm, cityInput, empty, cities, card, grid, render, cities"""


class GemmaLike(ScriptedModel):
    """Writes each section from the prompt; first script attempt is buggy; markup gets cut once."""

    def __init__(self):
        super().__init__(context_window=4096)
        self.script_attempts = 0
        self.markup_cut = False
        self.when_agent("fileplan", OUTLINE)
        self.when_agent("filesection", self._section)

    force_length = False

    def generate(self, messages, **kw):
        from dataclasses import replace
        r = super().generate(messages, **kw)
        if self.force_length:
            self.force_length = False
            r = replace(r, finish_reason="length")
        return r

    def _section(self, messages):
        body = messages[-1].content
        if "It was cut off" in body:
            return MARKUP[len(MARKUP) // 2:]
        m = re.search(r"WRITE NOW: section \d+ — ([^:]+):", body)
        name = m.group(1)
        if name == "head and css":
            return "```html\n" + HEAD + "```"            # model wraps in a fence; builder strips it
        if name == "markup":
            if not self.markup_cut:
                self.markup_cut = True
                self.force_length = True                    # simulate hitting the output budget
                return MARKUP[: len(MARKUP) // 2]
            return MARKUP
        self.script_attempts += 1
        return SCRIPT_BUGGY if self.script_attempts == 1 else SCRIPT_GOOD


def builder_and_tool(tmp_path, approve=True):
    model = GemmaLike()
    b = SectionedFileBuilder(model, ContextTracker(model, 0.9), section_tokens=300)
    writer = WriteFileTool(tmp_path / "out", StaticApprover(approve))
    return model, b, CreateFileTool(b, writer)


# ---------------------------------------------------------------- parsing
def test_plain_file_block():
    a = parse_action('Saving it.\nFILE: notes/todo.md\n# Todo\n- "quotes" and \\d are fine\nEND FILE\n')
    assert a.request.skill == "write_file" and a.request.arguments == {
        "path": "notes/todo.md", "content": '# Todo\n- "quotes" and \\d are fine\n'}


def test_file_block_without_end_is_refused():
    a = parse_action("FILE: x.html\n<html><body>")
    assert a.request is None and "END FILE" in a.error


def test_gemmas_json_habit_is_accepted():
    """Exactly the failures from real use: quotes in HTML and \\d in JS inside JSON."""
    raw = ('ACTION: write_file {"path": "w/weather.html", "content": "<div class="card">\\n'
           '<script>const re = /\\d+/;</script>\\n</div>"}')
    a = parse_action(raw)
    assert a.request.arguments["path"] == "w/weather.html"
    assert a.request.arguments["content"] == '<div class="card">\n<script>const re = /\\d+/;</script>\n</div>'


def test_outline_parsing_and_fallback():
    secs, names = parse_outline(OUTLINE, ".html")
    assert [s.name for s in secs] == ["head and css", "markup", "script"] and "cityInput" in names
    secs, _ = parse_outline("I will write the page now", ".html")
    assert len(secs) == 3                                   # sensible default plan for web pages
    assert [s.name for s in parse_outline("???", ".py")[0]] == ["whole file"]


# ------------------------------------------------------------ the builder
def test_build_in_sections_with_cutoff_and_targeted_fix(tmp_path):
    model, b, tool = builder_and_tool(tmp_path)
    obs = tool.invoke({"path": "weather_tracker/weather.html", "description": "beautiful weather tracker"},
                      task_id="t", purpose="Build a weather tracker")
    page = tmp_path / "out" / "weather_tracker" / "weather.html"
    assert obs.status == "success", obs.summary()
    assert page.read_text() == HEAD + MARKUP + SCRIPT_GOOD          # exact assembly, fences stripped, cut-off joined
    assert validate_file("weather.html", page.read_text()) == []
    assert not (page.parent / "weather.html.partial").exists()
    assert obs.result["bug_fix_rounds"] == 1 and obs.result["continuations"] >= 1
    assert model.script_attempts == 2                               # only the buggy section was rewritten
    # no section prompt ever contained the whole file — only the last lines
    section_prompts = [c.messages[-1].content for c in model.calls if c.messages[0].content.startswith("Stage: filesection.")]
    from tessera.tools.file_builder import TAIL_LINES
    for p in section_prompts:
        if "THE FILE SO FAR ENDS WITH:" in p:
            tail = p.split("THE FILE SO FAR ENDS WITH:\n", 1)[1].split("\n\nWRITE NOW", 1)[0]
            assert len(tail.splitlines()) <= TAIL_LINES
    fix_prompt = [p for p in section_prompts if "HAD THESE BUGS" in p][0]
    assert "JavaScript" in fix_prompt


def test_unfixable_file_is_not_published(tmp_path):
    model, b, tool = builder_and_tool(tmp_path)
    model.script_attempts = -100                                    # stays buggy forever
    orig = model._section
    model.rules[1] = lambda msgs: SCRIPT_BUGGY if "section 3" in msgs[-1].content else orig(msgs)
    obs = tool.invoke({"path": "w/weather.html", "description": "x"}, task_id="t", purpose="")
    assert obs.status == "error" and "NOT saved" in obs.stderr
    assert not (tmp_path / "out" / "w" / "weather.html").exists()
    assert (tmp_path / "out" / "w" / "weather.html.partial").exists()


def test_overwrite_needs_approval(tmp_path):
    (tmp_path / "out" / "w").mkdir(parents=True)
    (tmp_path / "out" / "w" / "weather.html").write_text("old")
    _, _, tool = builder_and_tool(tmp_path, approve=False)
    obs = tool.invoke({"path": "w/weather.html", "description": "x"}, task_id="t", purpose="")
    assert obs.status == "denied" and (tmp_path / "out" / "w" / "weather.html").read_text() == "old"


def test_end_to_end_develop_delegates_to_create_file(tmp_path):
    cfg = Config(database_path=str(tmp_path / "db.sqlite"), working_directory=str(tmp_path / "w"),
                 skill_directory=str(tmp_path / "g"), checkpoint_directory=str(tmp_path / "ck"),
                 output_directory=str(tmp_path / "out"), embedder="hashing", gui_enabled=False)
    from test_patch5_structure import ReplayGemma
    model = ReplayGemma()
    develop = iter(['ACTION: create_file {"path": "weather_tracker/weather.html", "description": "beautiful weather tracker with Open-Meteo"}',
                    "Created weather_tracker/weather.html."])
    (model.when_agent("router", "Complex|yes").when_agent("interpret", "INTENT: weather page")
          .when_agent("develop", lambda m: next(develop)).when_agent("evaluate", "VERDICT: PASS")
          .when_agent("respond", "Your weather tracker is saved."))
    system = build_system(cfg, model, StaticApprover(False), embedder=HashingEmbedder())
    reply = system.app.orchestrator.handle("Build a beautiful weather tracker page")
    assert (tmp_path / "out" / "weather_tracker" / "weather.html").exists()
    assert reply.text == "Your weather tracker is saved."
    dev = [c for c in model.calls if c.messages[0].content.startswith("Stage: develop.")]
    assert "create_file(" in dev[0].messages[-1].content and "FILE: folder/name.txt" in dev[0].messages[0].content
