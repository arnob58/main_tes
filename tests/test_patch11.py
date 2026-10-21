"""Patch 11 — from the real chat.html run: `const chat-history = …` failed 12 fix rounds
because Gemma only saw "Missing initializer in const declaration (line 1)"."""

from pathlib import Path

from tessera.tools.js_repair import explain, repair_js
from tessera.tools.validators import js_problems, validate_file

REAL = (Path(__file__).parent / "fixtures" / "hyphen_names_from_gemma.js").read_text()


def test_real_gemma_script_becomes_valid():
    assert js_problems(REAL)                       # broken as Gemma wrote it
    fixed, notes = repair_js(REAL)
    assert js_problems(fixed) == [], js_problems(fixed)
    assert "chatHistory.appendChild" in fixed and notes


def test_strings_and_selectors_untouched():
    code = "const chat-history = document.querySelector('.chat-history');\n// chat-history here\nchat-history.x = `a-b ${1}`;\n"
    fixed, _ = repair_js(code)
    assert "querySelector('.chat-history')" in fixed and "// chat-history here" in fixed
    assert fixed.startswith("const chatHistory") and "chatHistory.x" in fixed


def test_subtraction_is_not_renamed():
    code = "const total = 5;\nconst used = 2;\nconst left = total-used;\n"
    assert repair_js(code) == (code, [])


def test_hyphenated_params_and_functions():
    fixed, _ = repair_js("function add-message(the-text) { return the-text; }\nadd-message('x');\n")
    assert js_problems(fixed) == [] and "addMessage('x')" in fixed


def test_error_shows_the_line_and_a_hint():
    msg = js_problems("const a = 1;\nconst chat-history = 2;\n")[0]
    assert ">>    2 | const chat-history = 2;" in msg and "^" in msg
    assert "hyphens" in msg


def test_python_error_shows_line_and_hint():
    msg = validate_file("x.py", "def f():\nreturn 1\n")[0]
    assert ">>" in msg and "return 1" in msg and "indented" in msg


def test_explain_unknown_is_empty():
    assert explain("Something weird") == ""


def test_page_builder_repairs_hyphen_script_without_a_fix_round(tmp_path):
    from test_patch5_structure import ReplayGemma, tool
    model = ReplayGemma()
    model.script = "const city-input = document.getElementById('cityInput');\ncity-input.focus();\n"
    obs = tool(tmp_path, model).invoke({"path": "w/w.html", "description": "page"}, task_id="t", purpose="")
    assert obs.status == "success", obs.summary()
    assert "cityInput.focus()" in (tmp_path / "out" / "w" / "w.html").read_text()
    assert model.parts.count("script") == 1
