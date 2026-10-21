"""Phase 3: skill discovery, execution, observation, generation, recovery.

These run real child Python processes. Model output is scripted: the
"generated" code below is written by the test, standing in for Gemma.
"""

from pathlib import Path

import pytest

from tessera.app import build_app, build_skill_engine
from tessera.config import Config
from tessera.core.approval import StaticApprover
from tessera.core.types import SkillRequest
from tessera.execution.process_runner import ProcessRunner
from tessera.llm.scripted import ScriptedModel
from tessera.skills.contract import ContractError, parse_skill_source
from tessera.skills.policy import risky_operations

from conftest import stages_called, standard_model

EMPTY = Path("/nonexistent-builtin")


def cfg_for(tmp_path, **kw) -> Config:
    return Config(
        skill_directory=str(tmp_path / "generated"),
        working_directory=str(tmp_path / "work"),
        skill_timeout_s=20,
        **kw,
    )


def code_reply(src: str) -> str:
    return f"Here is the module.\n```python\n{src}```"


OBSERVE_SKILL = '''
SKILL = {"name": "observe_me", "description": "prints, returns and writes a file", "parameters": {"n": "int"}}
import sys
def run(n=1, **_):
    print("hello from stdout")
    print("warning on stderr", file=sys.stderr)
    with open("out.txt", "w") as fh:
        fh.write("x" * n)
    return {"doubled": n * 2}
'''


# --------------------------------------------------------------- contract
def test_contract_reads_metadata_without_importing(tmp_path):
    marker = tmp_path / "ran.txt"
    src = f'open({str(marker)!r}, "w").write("x")\nDESCRIPTION = "d"\ndef run(**k): return 1\n'
    spec = parse_skill_source(src, default_name="side_effect")
    assert spec.name == "side_effect" and not marker.exists()


@pytest.mark.parametrize("src, msg", [
    ("DESCRIPTION='x'\n", "run"),
    ("def run(**k): pass\n", "DESCRIPTION"),
    ("SKILL = make()\ndef run(): pass\n", "literal"),
    ("def run(:\n", "syntax"),
    ("DESCRIPTION='x'\nasync def run(): pass\n", "async"),
])
def test_contract_rejects(src, msg):
    with pytest.raises(ContractError, match=msg):
        parse_skill_source(src, default_name="x")


# ------------------------------------------------------- discovery (Test 3)
def test_dropped_module_is_discovered_without_code_changes(tmp_path):
    cfg = cfg_for(tmp_path)
    engine = build_skill_engine(cfg, ScriptedModel(), StaticApprover(), builtin_dir=EMPTY)
    assert len(engine.registry) == 0
    gen = Path(cfg.skill_directory)
    gen.mkdir(parents=True)
    (gen / "weather.py").write_text('DESCRIPTION = "Weather for a city"\ndef run(city): return {"city": city, "sky": "clear"}\n')
    (gen / "broken.py").write_text("def nope(:\n")
    assert engine.registry.refresh() == ["weather"]
    assert "weather" in engine.describe_capabilities()
    assert "broken.py" in next(iter(engine.registry.invalid))
    obs = engine.execute(SkillRequest("weather", {"city": "Toronto"}))
    assert obs.status == "success" and obs.result == {"city": "Toronto", "sky": "clear"}


def test_builtins_are_discovered():
    engine = build_skill_engine(Config(), ScriptedModel(), StaticApprover())
    names = {s.name for s in engine.registry.all()}
    assert {"calculator", "current_time"} <= names


# ------------------------------------------------------ observation (Test 7)
def test_runner_captures_everything(tmp_path):
    skill = tmp_path / "observe_me.py"
    skill.write_text(OBSERVE_SKILL)
    r = ProcessRunner(timeout_s=20).run(skill, {"n": 3}, tmp_path / "ws")
    assert r.ok and r.result == {"doubled": 6} and r.exit_code == 0
    assert r.stdout.strip() == "hello from stdout"
    assert "warning on stderr" in r.stderr
    assert [Path(a).name for a in r.artifacts] == ["out.txt"]
    assert r.duration_ms > 0


def test_runner_exception_is_not_success(tmp_path):
    skill = tmp_path / "boom.py"
    skill.write_text('DESCRIPTION="x"\ndef run(**k):\n    raise ValueError("bad input")\n')
    r = ProcessRunner(timeout_s=20).run(skill, {}, tmp_path)
    assert not r.ok and r.error_type == "ValueError" and "bad input" in r.message and "Traceback" in r.traceback


def test_runner_exit_zero_without_result_is_not_success(tmp_path):
    skill = tmp_path / "quitter.py"
    skill.write_text('import os\nDESCRIPTION="x"\ndef run(**k):\n    os._exit(0)\n')
    r = ProcessRunner(timeout_s=20).run(skill, {}, tmp_path)
    assert r.exit_code == 0 and not r.ok and r.error_type == "HarnessError"


def test_runner_timeout(tmp_path):
    skill = tmp_path / "slow.py"
    skill.write_text('import time\nDESCRIPTION="x"\ndef run(**k):\n    time.sleep(5)\n')
    r = ProcessRunner(timeout_s=0.5).run(skill, {}, tmp_path)
    assert r.timed_out and not r.ok and r.error_type == "TimeoutError"


# ------------------------------------------------------- resolution levels
def test_level1_builtin_needs_no_model(tmp_path):
    model = ScriptedModel()
    engine = build_skill_engine(cfg_for(tmp_path), model, StaticApprover())
    obs = engine.execute(SkillRequest("calculator", {"expression": "2**10 / sqrt(16)"}))
    assert obs.status == "success" and obs.result["value"] == 256.0
    assert model.calls == []


def test_level2_recipe_needs_no_model_and_is_promoted(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "Arnob_Resume.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "docs" / "notes.txt").write_text("hi")
    model = ScriptedModel()
    cfg = cfg_for(tmp_path)
    engine = build_skill_engine(cfg, model, StaticApprover(), builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("search_files", {"root": str(tmp_path / "docs"), "name_contains": "resume"}))
    assert obs.status == "success", obs.summary()
    assert [Path(m["path"]).name for m in obs.result["matches"]] == ["Arnob_Resume.pdf"]
    assert model.calls == []
    assert any("recipe 'find_files'" in n for n in obs.notes)
    assert (Path(cfg.skill_directory) / "find_files.py").exists()
    assert engine.registry.get("find_files").origin == "generated"


# --------------------------------------------- zero-skill bootstrap (Test 4)
GENERATED = '''import os
SKILL = {"name": "count_lines", "description": "Count lines in a text file", "parameters": {"path": "str"}}
def run(path, **_):
    with open(os.path.expanduser(path)) as fh:
        return {"lines": sum(1 for _ in fh)}
'''


def test_zero_skill_bootstrap_end_to_end(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("a\nb\nc\n")
    cfg = cfg_for(tmp_path, recipes_enabled=False)
    develop = iter([f'ACTION: count_lines {{"path": "{target}"}}', "The file has 3 lines."])
    model = (
        standard_model("Complex|yes")
        .when_agent("skillgen", code_reply(GENERATED))
    )
    model.when_agent("develop", lambda m: next(develop))
    model.rules.insert(0, model.rules.pop())
    engine = build_skill_engine(cfg, model, StaticApprover(), builtin_dir=EMPTY)
    assert len(engine.registry) == 0

    app = build_app(cfg, model, bridge=engine)
    app.orchestrator.handle("how many lines are in data.txt?")

    assert stages_called(model) == ["router", "interpret", "develop", "skillgen", "develop", "evaluate", "respond"]
    second_develop = model.calls[4].messages[-1].content
    assert "status=success" in second_develop and "'lines': 3" in second_develop
    assert "generated new skill 'count_lines'" in second_develop
    assert engine.registry.get("count_lines") is not None          # reusable now
    # second request: Level 1, no generation call
    n = len(model.calls)
    obs = engine.execute(SkillRequest("count_lines", {"path": str(target)}))
    assert obs.result == {"lines": 3} and len(model.calls) == n


def test_generation_vs_execution_distinguished(tmp_path):
    model = ScriptedModel().when_agent("skillgen", "I can't write that.")
    engine = build_skill_engine(cfg_for(tmp_path, recipes_enabled=False), model, StaticApprover(), builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("teleport", {}))
    assert obs.status == "error" and "nothing was executed" in obs.notes[0]
    assert len(model.calls) == 2  # one retry with the validation error


# ------------------------------------------------ missing dependency (Test 5)
NEEDS_DEP = '''import tessera_fake_dep
SKILL = {"name": "use_dep", "description": "uses a third-party package", "dependencies": ["tessera-fake-dep"]}
def run(**_):
    return tessera_fake_dep.VALUE
'''


def test_missing_dependency_is_installed_and_retried(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setenv("TESSERA_SKILL_PATHS", str(site))
    installed = []

    def fake_pip(package):
        installed.append(package)
        (site / "tessera_fake_dep.py").write_text("VALUE = 42\n")
        return True, "ok"

    approver = StaticApprover(True)
    model = ScriptedModel().when_agent("skillgen", code_reply(NEEDS_DEP))
    engine = build_skill_engine(cfg_for(tmp_path, recipes_enabled=False), model, approver, installer=fake_pip, builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("use_dep", {}))
    assert obs.status == "success" and obs.result == 42
    assert installed == ["tessera_fake_dep"]
    assert approver.asked[0][0] == "install_package"
    job = engine.jobs[-1]
    assert job.attempt == 2 and job.dependencies == ["tessera_fake_dep"]


def test_missing_dependency_denied_reports_accurately(tmp_path):
    model = ScriptedModel().when_agent("skillgen", code_reply(NEEDS_DEP))
    engine = build_skill_engine(cfg_for(tmp_path, recipes_enabled=False), model, StaticApprover(False),
                                installer=lambda p: pytest.fail("must not install"), builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("use_dep", {}))
    assert obs.status == "error" and "ModuleNotFoundError" in obs.stderr
    assert any("not approved" in n for n in obs.notes)
    assert not (Path(cfg_for(tmp_path).skill_directory) / "use_dep.py").exists()


# ------------------------------------------------------------- repair loop
BUGGY = '''SKILL = {"name": "average", "description": "Average of numbers", "parameters": {"values": "list[float]"}}
def run(values, **_):
    return sum(values) / len(value)
'''
FIXED = BUGGY.replace("len(value)", "len(values)")


def test_novel_error_escalates_to_model_repair(tmp_path):
    model = ScriptedModel().when_agent("skillgen", code_reply(BUGGY)).when_agent("skillfix", code_reply(FIXED))
    engine = build_skill_engine(cfg_for(tmp_path, recipes_enabled=False), model, StaticApprover(), builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("average", {"values": [1, 2, 3]}))
    assert obs.status == "success" and obs.result == 2.0
    fix_call = [c for c in model.calls if c.messages[0].content.startswith("Stage: skillfix.")][0]
    assert "NameError" in fix_call.messages[1].content
    assert any("repaired by the model after NameError" in n for n in obs.notes)


def test_identical_failure_stops_retrying(tmp_path):
    model = ScriptedModel().when_agent("skillgen", code_reply(BUGGY)).when_agent("skillfix", code_reply(BUGGY))
    engine = build_skill_engine(cfg_for(tmp_path, recipes_enabled=False, max_skill_retries=5), model,
                                StaticApprover(), builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("average", {"values": [1]}))
    assert obs.status == "error" and "NameError" in obs.stderr
    assert engine.jobs[-1].attempt == 3  # stopped on the third identical failure, not 6


def test_unexpected_argument_fixed_mechanically(tmp_path):
    model = ScriptedModel()
    engine = build_skill_engine(cfg_for(tmp_path), model, StaticApprover(), builtin_dir=EMPTY)
    gen = Path(cfg_for(tmp_path).skill_directory)
    gen.mkdir(parents=True)
    (gen / "greet.py").write_text('DESCRIPTION="Greets"\ndef run(name):\n    return "hi " + name\n')
    obs = engine.execute(SkillRequest("greet", {"name": "Ana", "tone": "warm"}))
    assert obs.status == "success" and obs.result == "hi Ana" and model.calls == []
    assert any("dropped unexpected argument 'tone'" in n for n in obs.notes)


# ---------------------------------------------------------------- safety
def test_risky_code_requires_approval(tmp_path):
    risky = 'import shutil\nSKILL={"name":"wipe","description":"removes a folder"}\ndef run(path, **_):\n    shutil.rmtree(path)\n'
    victim = tmp_path / "keep"
    victim.mkdir()
    approver = StaticApprover(False)
    model = ScriptedModel().when_agent("skillgen", code_reply(risky))
    engine = build_skill_engine(cfg_for(tmp_path, recipes_enabled=False), model, approver, builtin_dir=EMPTY)
    obs = engine.execute(SkillRequest("wipe", {"path": str(victim)}))
    assert obs.status == "denied" and victim.exists()
    assert approver.asked == [("run_risky_code", "wipe: shutil.rmtree()")]


def test_policy_scan():
    assert risky_operations("import os\nos.remove('x')\nPath('y').unlink()") == [".unlink()", "os.remove()"]
    assert risky_operations("print(1)") == []


# ------------------------------------------- dynamic reasoning (Test 8)
def test_pipeline_find_then_read_with_recipes(tmp_path):
    desk = tmp_path / "Desktop"
    desk.mkdir()
    (desk / "my_resume.txt").write_text("Arnob — Social Media Coordinator. Skills: content strategy.")
    cfg = cfg_for(tmp_path)
    develop = iter([
        f'I will search first.\nACTION: find_files {{"root": "{desk}", "name_contains": "resume"}}',
        f'ACTION: read_text_file {{"path": "{desk / "my_resume.txt"}"}}',
        "The resume belongs to Arnob, a Social Media Coordinator.",
    ])
    model = standard_model("Complex|yes")
    model.when_agent("develop", lambda m: next(develop))
    model.rules.insert(0, model.rules.pop())
    engine = build_skill_engine(cfg, model, StaticApprover(), builtin_dir=EMPTY)
    reply = build_app(cfg, model, bridge=engine).orchestrator.handle("find my resume on my desktop and read it")
    respond_prompt = model.calls[-1].messages[-1].content
    assert "content strategy" in respond_prompt and "my_resume.txt" in respond_prompt
    assert "skillgen" not in stages_called(model)
    assert reply.task["counters"]["skill_calls"] == 2


def test_relative_working_directory_regression(tmp_path, monkeypatch):
    """Found by the end-to-end console run: relative workspaces broke the child's arg file path."""
    monkeypatch.chdir(tmp_path)
    cfg = Config(skill_directory="gen", working_directory="storage/working")
    engine = build_skill_engine(cfg, ScriptedModel(), StaticApprover())
    obs = engine.execute(SkillRequest("calculator", {"expression": "6*7"}, task_id="t1"))
    assert obs.status == "success" and obs.result["value"] == 42, obs.summary()
