"""create_file: build a file of any size section by section, never holding it all in context.

Flow (each model call is a fresh, small prompt):
  1. outline   — sections in file order + the SHARED NAMES every section must use
                 (CSS classes, element ids, function names) so the parts fit together
  2. sections  — one call per section with: task, outline, shared names, and only the
                 last few lines already on disk. Output is raw text (no JSON, no
                 escaping) appended to `<file>.partial` immediately.
  3. cut-offs  — a section that hits its output budget is continued from the tail on
                 disk in a fresh context (same idea as Phase 4 recovery)
  4. validate  — the assembled file is checked; a problem is mapped to the section that
                 contains it and only that section is regenerated, with the problems
  5. publish   — only a file that passes validation replaces the target; otherwise the
                 `.partial` is kept for inspection and the failure is reported

SRP: orchestration of the steps above. Validation lives in validators.py, path rules
and approvals in files.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..context.continuation import merge_continuation, strip_fences
from ..context.tracker import ContextTracker
from ..core.events import NullEvents
from ..core.interfaces import EventSink, ModelProvider
from ..core.types import Message, Observation
from ..skills.contract import SkillSpec
from .files import WriteFileTool
from .html_builder import HtmlPageBuilder, repair_css
from ..skills.policy import risky_operations
from .dedupe import dedupe_sections
from .py_checks import import_problems, reorder_entrypoints
from .requirements import requirement_problems
from .section_checks import language_for, redefinitions
from .validators import validate_file

OUTLINE_PROMPT = """Stage: fileplan.
Plan a file that will be written in sections, in order from top to bottom.
Reply in exactly this plain-text format, nothing else:
SECTION: <short name> | <what this part contains>
SECTION: ...
NAMES: <comma-separated ids, CSS classes, functions and storage keys that several sections must share>
Use 2 to 6 sections. For a web page, typical sections are: document head and CSS, page markup, JavaScript."""

SECTION_PROMPT = """Stage: filesection.
You are writing ONE section of the file {path}. Output ONLY the raw text of this section, exactly as it
should appear in the file — no markdown fences, no explanations, no other sections.
Write {unit} only: never stop in the middle of one. Reuse names that already exist; never define them again.
Write ONLY this section's part — never the whole program. Never fake or mock a missing module.
Keep it focused: about 120 lines at most."""

TAIL_LINES = 15
_SECTION_RE = re.compile(r"^\s*SECTION\s*:\s*(.+?)\s*(?:\|\s*(.+))?$", re.I | re.M)
_NAMES_RE = re.compile(r"^\s*NAMES\s*:\s*(.+)$", re.I | re.M)
_LINE_RE = re.compile(r"line (\d+)")
_START_RE = re.compile(r"\[(?:script|style) starting at line (\d+)\]")

DEFAULT_SECTIONS = {
    ".html": [("head and css", "doctype, <head> with title and all CSS in a <style> tag"),
              ("markup", "<body> opening tag and all visible page markup"),
              ("script", "one <script> with all JavaScript, then </body></html>")],
}


@dataclass
class Section:
    name: str
    purpose: str
    text: str = ""
    start_line: int = 1
    end_line: int = 0


@dataclass
class BuildResult:
    ok: bool
    path: Path
    sections: list[Section]
    problems: list[str] = field(default_factory=list)
    model_calls: int = 0
    fixes: int = 0
    continuations: int = 0
    notes: list[str] = field(default_factory=list)
    resume: dict = field(default_factory=dict)


def parse_outline(text: str, suffix: str) -> tuple[list[Section], str]:
    sections = [Section(m.group(1).strip()[:60], (m.group(2) or m.group(1)).strip()[:300])
                for m in _SECTION_RE.finditer(text)]
    names = (_NAMES_RE.search(text).group(1).strip() if _NAMES_RE.search(text) else "")
    if not 1 <= len(sections) <= 8:
        defaults = DEFAULT_SECTIONS[".html"] if suffix in (".html", ".htm") else None
        sections = [Section(n, p) for n, p in defaults] if defaults else [Section("whole file", "the complete file")]
    return sections, names


def section_for_problem(problem: str, sections: list[Section]) -> Section:
    m = _START_RE.search(problem)
    if m:
        line = int(m.group(1))
        rel = _LINE_RE.search(problem[: m.start()])
        if rel and "JavaScript" in problem:
            line += int(rel.group(1)) - 1
    else:
        m2 = _LINE_RE.search(problem)
        line = int(m2.group(1)) if m2 else sections[-1].end_line
    for s in sections:
        if s.start_line <= line <= s.end_line:
            return s
    return sections[-1]


def _section_for_requirement(problem: str, sections: list[Section]) -> Section:
    """Send a missing-requirement problem to the section whose plan best matches the missing names."""
    words = set(re.findall(r"[a-z]+", problem.split("but the file")[0].lower())) - {"the", "request", "asks", "for"}
    best = max(sections, key=lambda s: len(words & set(re.findall(r"[a-z]+", f"{s.name} {s.purpose}".lower()))))
    return best if words & set(re.findall(r"[a-z]+", f"{best.name} {best.purpose}".lower())) else sections[-1]


class SectionedFileBuilder:
    """Generic builder for any non-HTML file. Every section is checked the moment it
    is written and fixed before the next one starts; later sections receive the
    names earlier sections *actually* defined (read deterministically)."""

    def __init__(self, model: ModelProvider, tracker: ContextTracker, *, events: EventSink | None = None,
                 section_tokens: int = 1200, max_continuations: int = 3, max_fix_rounds: int = 3,
                 section_attempts: int = 2, repeat_penalty: float = 1.12,
                 import_check: bool = True, import_paths: tuple = ()) -> None:
        self._import_check = import_check
        self._import_paths = tuple(import_paths) or (Path.cwd(),)
        self._model = model
        self._tracker = tracker
        self._events = events or NullEvents()
        self._section_tokens = section_tokens
        self._max_cont = max_continuations
        self._max_fix = max_fix_rounds
        self._section_attempts = section_attempts
        self._repeat_penalty = repeat_penalty
        self.calls = 0

    def _gen(self, system: str, user: str, cap: int):
        msgs = [Message.system(system), Message.user(user)]
        budget = min(cap, self._tracker.measure(msgs).safe_output_budget)
        if budget < 32:
            raise RuntimeError("section prompt leaves no room to write; shorten the description")
        self.calls += 1
        return self._model.generate(msgs, max_output_tokens=budget, temperature=0.3,
                                    repeat_penalty=self._repeat_penalty)

    def build(self, target: Path, description: str, objective: str = "",
              previous: dict | None = None, extra_check=None) -> BuildResult:
        self._extra_check = extra_check  # Callable[[str], list[str]] — e.g. requirement checks
        partial = target.with_name(target.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        task = description if not objective or objective in description else f"{description}\n\nUser's request: {objective}"
        lang = language_for(target.suffix)

        if previous and previous.get("sections"):
            sections = [Section(n, p, t) for n, p, t in previous["sections"]]
            names = previous.get("names", "")
            kept = sum(1 for s in sections if s.text)
            self._events.emit("File", f"{target.name}: resuming — keeping {kept} of {len(sections)} sections")
        else:
            r = self._gen(OUTLINE_PROMPT, f"FILE: {target.name}\nWHAT IT MUST DO:\n{task}", 400)
            sections, names = parse_outline(r.text, target.suffix.lower())
            self._events.emit("File", f"{target.name}: {len(sections)} sections planned")
        outline = "\n".join(f"{i + 1}. {s.name}: {s.purpose}" for i, s in enumerate(sections))

        res = BuildResult(False, target, sections)
        for i, s in enumerate(sections):
            if s.text:
                continue  # passed in an earlier attempt
            problems = ""
            for attempt in range(self._section_attempts):
                s.text, cont = self._write_section(target, task, outline, names, sections, i, problems, lang)
                res.continuations += cont
                own = self._section_problems(s, lang, res)
                if not own:
                    break
                problems = "\n".join(own)
                self._events.emit("File", f"section {i + 1} '{s.name}': {own[0][:90]} — rewriting")
            self._flush(partial, sections)
            self._events.emit("File", f"section {i + 1}/{len(sections)} '{s.name}' written ({s.text.count(chr(10))} lines)")

        problems = self._whole_problems(target, partial, sections, lang, res)
        while problems and res.fixes < self._max_fix:
            res.fixes += 1
            bad, related = problems[0]
            i = sections.index(bad)
            self._events.emit("File", f"fix {res.fixes}/{self._max_fix}: section '{bad.name}' — {related[0][:90]}")
            bad.text, cont = self._write_section(target, task, outline, names, sections, i, "\n".join(related), lang)
            res.continuations += cont
            self._section_problems(bad, lang, res)
            problems = self._whole_problems(target, partial, sections, lang, res)

        res.problems = [p for _, ps in problems for p in ps]
        res.model_calls = self.calls
        bad_ids = {id(s) for s, _ in problems}
        res.resume = {"sections": [(s.name, s.purpose, "" if id(s) in bad_ids else s.text) for s in sections],
                      "names": names}
        if not res.problems:
            os.replace(partial, target)
            res.ok = True
        return res

    # ------------------------------------------------------------------ checks
    def _section_problems(self, s: Section, lang, res: BuildResult) -> list[str]:
        if lang.label == "CSS":
            fixed, notes = repair_css(s.text)
            if notes:
                s.text = fixed
                res.notes += notes
        if lang.label == "JavaScript":
            from .js_repair import repair_js

            fixed, notes = repair_js(s.text)
            if notes:
                s.text = fixed
                res.notes += notes
        return lang.check(s.text)

    def _whole_problems(self, target, partial, sections, lang, res) -> list[tuple[Section, list[str]]]:
        texts, notes = dedupe_sections([s.name for s in sections], [s.text for s in sections], target.suffix)
        if notes:
            for s, t in zip(sections, texts):
                s.text = t if t.endswith("\n") or not t else t + "\n"
            res.notes += notes
            self._events.emit("File", "duplicates removed: " + "; ".join(notes)[:160])
        if target.suffix.lower() == ".py":
            texts, moved = reorder_entrypoints([s.text for s in sections])
            if moved:
                for s, t in zip(sections, texts):
                    s.text = t
                res.notes += moved
                self._events.emit("File", "; ".join(moved))
        data = self._flush(partial, sections)
        grouped: dict[int, list[str]] = {}
        for p in validate_file(target.name, data):
            grouped.setdefault(sections.index(section_for_problem(p, sections)), []).append(p)
        for i, ps in redefinitions([s.text for s in sections], lang).items():
            grouped.setdefault(i, []).extend(ps)
        if not grouped and getattr(self, "_extra_check", None):
            for p in self._extra_check(data):
                grouped.setdefault(sections.index(_section_for_requirement(p, sections)), []).append(p)
        if not grouped and target.suffix.lower() == ".py" and self._import_check:
            if risky_operations(data):
                res.notes.append("import test skipped: the file uses operations that need your approval to run")
            else:
                self._events.emit("File", f"{target.name}: test-importing the finished file…")
                blocking, warnings = import_problems(partial.with_name(target.name + ".partial"),
                                                     self._import_paths)
                res.notes += warnings
                for p in blocking:
                    grouped.setdefault(sections.index(section_for_problem(p, sections)), []).append(p)
        return [(sections[i], ps) for i, ps in sorted(grouped.items())]

    # ------------------------------------------------------------------- steps
    def _write_section(self, target, task, outline, names, sections, i, problems, lang) -> tuple[str, int]:
        s = sections[i]
        before = "".join(x.text for x in sections[:i])
        tail = "\n".join(before.rstrip("\n").splitlines()[-TAIL_LINES:]) if before else "(this is the start of the file)"
        after = ", ".join(x.name for x in sections[i + 1:]) or "nothing — this is the end of the file"
        defined = [d for x in sections[:i] for d in lang.defines(x.text)]
        user = (f"TASK:\n{task}\n\nOUTLINE:\n{outline}\n\nSHARED NAMES: {names or '(choose clear names)'}\n\n"
                + ("ALREADY DEFINED IN EARLIER SECTIONS (use these, do not redefine):\n" + "\n".join(defined[:60]) + "\n\n"
                   if defined else "")
                + f"THE FILE SO FAR ENDS WITH:\n{tail}\n\n"
                f"WRITE NOW: section {i + 1} — {s.name}: {s.purpose}\nAFTER THIS COMES: {after}")
        if problems:
            user += f"\n\nYOUR PREVIOUS VERSION OF THIS SECTION HAD THESE BUGS — fix them:\n{problems}"
        system = SECTION_PROMPT.format(path=target.name, unit=lang.unit)
        r = self._gen(system, user, self._section_tokens)
        text, conts = _clean(r.text, finished=r.finish_reason != "length"), 0
        while r.finish_reason == "length" and conts < self._max_cont:
            conts += 1
            tail2 = "\n".join(text.splitlines()[-TAIL_LINES:])
            last = ([l for l in text.splitlines() if l.strip()] or [""])[-1]
            r = self._gen(system,
                          f"TASK:\n{task}\n\nYou were writing section {i + 1} — {s.name}: {s.purpose}\n"
                          f"It was cut off. It currently ends with:\n{tail2}\n\nThe very last line was:\n{last}\n"
                          "Continue EXACTLY after that line. Do not repeat anything already written.", self._section_tokens)
            # A cut can land mid-line: join raw text, never insert a line break at the seam.
            text = merge_continuation(text, _clean(r.text, finished=r.finish_reason != "length", head=False))
        return (text if text.endswith("\n") else text + "\n"), conts

    @staticmethod
    def _flush(partial: Path, sections: list[Section]) -> str:
        line = 1
        for s in sections:
            s.start_line = line
            line += s.text.count("\n")
            s.end_line = max(s.start_line, line - 1)
        data = "".join(s.text for s in sections)
        partial.write_text(data, encoding="utf-8")
        return data


def _clean(text: str, *, finished: bool = True, head: bool = True) -> str:
    """Strip chat wrappers a small model may add around raw text.

    `finished=False` (cut off): keep the exact ending so the continuation can
    resume mid-line. `head=False` (a continuation): keep leading characters.
    """
    t = strip_fences(text.lstrip("\n") if head else text)
    if head:
        t = re.sub(r"^\s*(SECTION|WRITE NOW)[^\n]*\n", "", t, flags=re.I)
    return t.rstrip() + "\n" if finished else t


class CreateFileTool:
    spec = SkillSpec(
        name="create_file",
        description=("Create a complete file of any size (web page, script, stylesheet, document). It is written "
                     "for you in parts, each checked and fixed, then the whole file is checked. Give a detailed "
                     "description; don't write the content yourself. Calling it again for the same path resumes "
                     "and only redoes the parts that failed"),
        parameters={"path": "str - e.g. weather_tracker/weather.html",
                    "description": "str - everything the file must do and look like"},
        origin="native",
    )

    PAGE_TYPES = (".html", ".htm")

    def __init__(self, builder: SectionedFileBuilder, writer: WriteFileTool,
                 page_builder: HtmlPageBuilder | None = None, file_reader=None, reference_chars: int = 2500) -> None:
        self._builder, self._writer, self._pages = builder, writer, page_builder
        self._files = file_reader          # reads files named in the user's request, for reference
        self._ref_chars = reference_chars
        self._resume: dict[str, dict] = {}   # target path → good parts from a failed attempt

    def _references(self, request: str) -> str:
        """The writer sees the files the user named (e.g. __main__.py), trimmed to fit."""
        if self._files is None or not request:
            return ""
        parts, left = [], self._ref_chars
        for obs in self._files.read_mentioned(request):
            if obs.status != "success" or left <= 200:
                continue
            text = str(obs.result)[:left]
            left -= len(text)
            parts.append(f"--- {obs.source[5:]} ---\n{text}")
        return ("\n\nREFERENCE FILES the user named (follow how they do things):\n" + "\n".join(parts)) if parts else ""

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        raw = str(arguments.get("path") or "").strip()
        if not raw:
            return Observation("error", self.spec.name, stderr="missing 'path'")
        target = self._writer._target(raw)
        out = self._writer.output_dir
        key = str(target)
        if not (out == target or out in target.parents) and not self._writer._approver.approve(
                "write_outside_output", f"write {target}"):
            return Observation("denied", self.spec.name, result="writing outside the output folder was not approved")
        if key not in self._resume and target.exists() and not self._writer._approver.approve(
                "overwrite_file", f"overwrite existing {target}"):
            return Observation("denied", self.spec.name, result=f"{target} exists; overwrite not approved")
        description = str(arguments.get("description") or purpose) + self._references(purpose)
        task = description if not purpose or purpose in description else f"{description}\n\nUser's request: {purpose}"
        try:
            check = (lambda text, sfx=target.suffix: requirement_problems(text, purpose, sfx)) if purpose else None
            if self._pages is not None and target.suffix.lower() in self.PAGE_TYPES:
                res = self._pages.build(target, task, self._resume.get(key), extra_check=check)
                parts_info, resume = list(res.parts), res.good_parts
            else:
                res = self._builder.build(target, description, purpose, self._resume.get(key), extra_check=check)
                parts_info, resume = [s.name for s in res.sections], res.resume
        except RuntimeError as exc:
            return Observation("error", self.spec.name, stderr=str(exc))
        info = {"path": str(target), "parts": parts_info, "model_calls": res.model_calls,
                "bug_fix_rounds": res.fixes, "continuations": res.continuations}
        notes = tuple(getattr(res, "notes", []) or [])
        if res.ok:
            self._resume.pop(key, None)
            return Observation("success", self.spec.name, result=info, artifacts=(str(target),),
                               notes=(f"saved to {target} (passed validation)",) + notes)
        self._resume[key] = resume
        return Observation("error", self.spec.name, result=info,
                           stderr="File still has bugs after fixing; NOT saved:\n- " + "\n- ".join(res.problems[:6]),
                           artifacts=(str(target) + ".partial",),
                           notes=("calling create_file again for the same path resumes and keeps the parts that passed",)
                           + notes)


class AliasTool:
    """Another name for an existing tool (keeps old habits working, one implementation)."""

    def __init__(self, name: str, description: str, target) -> None:
        self.spec = SkillSpec(name=name, description=description, parameters=dict(target.spec.parameters),
                              origin="native")
        self._target = target

    def invoke(self, arguments, *, task_id, purpose):
        return self._target.invoke(arguments, task_id=task_id, purpose=purpose)
