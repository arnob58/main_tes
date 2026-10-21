"""Patch 5: Tessera owns file structure; names come from what was actually written; retries resume.

The HTML tests replay the *real* weather.html.partial from Gemma: a "head" part that
emitted a whole empty page, CSS that never closed </style>, duplicated CSS rules after
a cut-off, and CSS ids (#city-input) that don't match the markup (id="cityInput").
"""

from pathlib import Path

from tessera.context.tracker import ContextTracker
from tessera.core.approval import StaticApprover
from tessera.llm.scripted import ScriptedModel
from tessera.tools.file_builder import CreateFileTool, SectionedFileBuilder
from tessera.tools.files import WriteFileTool
from tessera.tools.html_builder import HtmlPageBuilder, check_parts, repair_css
from tessera.tools.validators import validate_file

REAL = (Path(__file__).parent / "fixtures" / "weather_partial_from_gemma.html").read_text().splitlines(keepends=True)
REAL_SKELETON = "".join(REAL[0:14])      # the "head" section that wrote a whole empty page
REAL_CSS = "".join(REAL[14:267])         # starts with <style>, never closes it, duplicates rules
REAL_MARKUP = "".join(REAL[267:285])     # ids: cityInput, addCityBtn, messageArea, weatherGrid

GOOD_SCRIPT = """const KEY = "cities";
const input = document.getElementById("cityInput");
const grid = document.getElementById("weatherGrid");
const message = document.getElementById("messageArea");
function load() { return JSON.parse(localStorage.getItem(KEY) || "[]"); }
function save(list) { localStorage.setItem(KEY, JSON.stringify(list)); }
function render() {
  const list = load();
  message.hidden = list.length > 0;
  grid.innerHTML = "";
  for (const city of list) {
    const card = document.createElement("div");
    card.className = "weather-card";
    card.innerHTML = `<span class="city-name">${city}</span>`;
    grid.append(card);
  }
}
document.getElementById("addCityBtn").addEventListener("click", () => {
  const c = input.value.trim();
  if (c) { save([...load(), c]); render(); }
});
render();
"""
FIXED_CSS = """.container { max-width: 1100px; margin: auto; }
.input-group { display: flex; gap: 10px; }
#cityInput { flex: 1; padding: 12px; border-radius: 8px; }
#addCityBtn { padding: 12px 20px; border-radius: 8px; }
.message-box, #messageArea { text-align: center; padding: 30px; }
.grid-container, #weatherGrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }
.weather-card { backdrop-filter: blur(12px); border-radius: 16px; padding: 20px; }
.weather-card:hover { transform: translateY(-4px); }
.city-name { font-weight: 600; }
"""


class ReplayGemma(ScriptedModel):
    """Answers each page part the way Gemma really did, then fixes when told what's wrong."""

    def __init__(self, script=GOOD_SCRIPT):
        super().__init__(context_window=32768)
        self.parts: list[str] = []
        self.script = script
        self.when_agent("filesection", self._part)

    def _part(self, messages):
        body = messages[-1].content
        part = "markup" if "the page markup" in body else "script" if "the JavaScript" in body else "css"
        self.parts.append(part)
        fixing = "HAD THESE PROBLEMS" in body
        if part == "markup":
            return REAL_SKELETON + REAL_MARKUP                   # whole-page habit, as observed
        if part == "script":
            return self.script
        return FIXED_CSS if fixing else REAL_CSS


def tool(tmp_path, model, approve=True):
    tracker = ContextTracker(model, 0.9)
    writer = WriteFileTool(tmp_path / "out", StaticApprover(approve))
    return CreateFileTool(SectionedFileBuilder(model, tracker), writer, HtmlPageBuilder(model, tracker))


# ------------------------------------------------------------------- HTML
def test_real_failure_modes_are_handled_and_page_is_valid(tmp_path):
    model = ReplayGemma()
    obs = tool(tmp_path, model).invoke({"path": "weather_tracker/weather.html", "description": "weather tracker"},
                                       task_id="t", purpose="Build a beautiful weather tracker")
    page = (tmp_path / "out" / "weather_tracker" / "weather.html")
    assert obs.status == "success", obs.summary()
    text = page.read_text()
    assert validate_file("weather.html", text) == []
    assert text.count("<!DOCTYPE") == 1 and text.count("<style>") == 1 and text.count("</style>") == 1
    assert text.count("<body>") == 1 and text.count("</html>") == 1
    assert "CSS will go here" not in text                      # the empty skeleton was stripped
    assert 'id="cityInput"' in text and "#cityInput" in text and "#city-input" not in text
    assert model.parts == ["markup", "script", "css", "css"]    # only the CSS was redone
    fix_prompt = [c.messages[-1].content for c in model.calls if "HAD THESE PROBLEMS" in c.messages[-1].content][0]
    assert "#city-input" in fix_prompt and "cityInput" in fix_prompt


def test_css_repair_trims_cutoff_and_duplicates():
    css, notes = repair_css(".a { color: red; }\n.b { x: 1; }\n.a { color: blue; }\n.c { width: 1")
    assert css.count(".a") == 1 and "blue" in css and ".c" not in css
    assert any("unfinished" in n for n in notes) and any("duplicate" in n for n in notes)


def test_cross_checks_name_mismatches():
    problems = check_parts({"markup": '<input id="cityInput">', "script": 'document.getElementById("city-input")',
                            "css": "#cityInput { color: red; }"})
    assert "script" in problems and "city-input" in problems["script"][0]
    assert "css" not in problems


def test_retry_resumes_and_keeps_passed_parts(tmp_path):
    broken_script = "function render( {\n"
    model = ReplayGemma(script=broken_script)
    t = tool(tmp_path, model)
    first = t.invoke({"path": "w/weather.html", "description": "x"}, task_id="t", purpose="")
    assert first.status == "error" and "resumes" in " ".join(first.notes)
    model.script = GOOD_SCRIPT
    before = list(model.parts)
    second = t.invoke({"path": "w/weather.html", "description": "x"}, task_id="t", purpose="")
    assert second.status == "success", second.summary()
    redone = model.parts[len(before):]
    assert "markup" not in redone                                 # kept from the first attempt


# ---------------------------------------------------------------- other files
PY_SECTIONS = {
    "helpers": "import json\n\ndef parse_city(raw):\n    return raw.strip().title()\n",
    "main_bad": "def main(:\n    pass\n",
    "main_redefines": "def parse_city(raw):\n    return raw\n\ndef main():\n    print(parse_city(' x '))\n",
    "main_good": "def main():\n    print(parse_city(' toronto '))\n\nif __name__ == '__main__':\n    main()\n",
}


class PyGemma(ScriptedModel):
    def __init__(self):
        super().__init__(context_window=32768)
        self.section2 = 0
        self.when_agent("fileplan", "SECTION: helpers | parsing helpers\nSECTION: main | entry point\nNAMES: parse_city, main")
        self.when_agent("filesection", self._section)

    def _section(self, messages):
        body = messages[-1].content
        if "section 1 —" in body:
            return PY_SECTIONS["helpers"]
        self.section2 += 1
        return [PY_SECTIONS["main_bad"], PY_SECTIONS["main_redefines"], PY_SECTIONS["main_good"]][min(self.section2 - 1, 2)]


def test_python_sections_checked_immediately_and_contract_passed(tmp_path):
    model = PyGemma()
    obs = tool(tmp_path, model).invoke({"path": "tools/cities.py", "description": "city helper"}, task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    src = (tmp_path / "out" / "tools" / "cities.py").read_text()
    assert src.count("def parse_city") == 1 and "def main():" in src
    compile(src, "cities.py", "exec")
    prompts = [c.messages[-1].content for c in model.calls if "section 2 —" in c.messages[-1].content]
    assert "def parse_city(raw)" in prompts[0]                  # contract from the code actually written
    assert "invalid syntax" in prompts[1] or "Python:" in prompts[1]   # fixed before moving on
    # the duplicate parse_city is removed by Tessera (section 'helpers' keeps it), not by asking Gemma
    assert not any("already defined in section 1" in p for p in prompts)
    assert any("kept 'parse_city'" in n for n in obs.notes)


def test_standalone_css_cutoff_repaired_without_model(tmp_path):
    model = ScriptedModel(context_window=32768)
    model.when_agent("fileplan", "SECTION: all | styles\nNAMES: card")
    model.when_agent("filesection", ".card { padding: 4px; }\n.card { padding: 8px; }\n.grid { display: grid;")
    obs = tool(tmp_path, model).invoke({"path": "s/site.css", "description": "styles"}, task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    css = (tmp_path / "out" / "s" / "site.css").read_text()
    assert css.count(".card") == 1 and "8px" in css and ".grid" not in css
    assert len([c for c in model.calls if c.messages[0].content.startswith("Stage: filesection.")]) == 1


def test_markdown_and_unknown_types_still_work(tmp_path):
    model = ScriptedModel(context_window=32768)
    model.when_agent("fileplan", "SECTION: intro | intro\nSECTION: usage | usage")
    model.when_agent("filesection", lambda m: "# Intro\n\nHello.\n" if "section 1 —" in m[-1].content else "## Usage\n\nRun it.\n")
    obs = tool(tmp_path, model).invoke({"path": "d/README.md", "description": "readme"}, task_id="t", purpose="")
    assert obs.status == "success"
    assert (tmp_path / "out" / "d" / "README.md").read_text() == "# Intro\n\nHello.\n## Usage\n\nRun it.\n"


def test_unused_css_class_never_blocks_saving(tmp_path):
    """From real use: '.temperature but no element uses that class' rejected a working page 8 times."""
    model = ReplayGemma()
    extra = ".temperature { font-size: 2em; }\n.city-input { color: red; }\n"
    model._part_orig = model._part
    model.rules[0] = lambda m: (FIXED_CSS + extra) if "the CSS" in m[-1].content else model._part_orig(m)
    obs = tool(tmp_path, model).invoke({"path": "w/weather.html", "description": "x"}, task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    assert any("styling warning" in n for n in obs.notes)
    css_calls = [c for c in model.calls if "the CSS" in c.messages[-1].content]
    assert len(css_calls) == 2                                  # written once + one tidy attempt, then saved
