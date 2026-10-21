"""Web pages built from three pure parts inside a skeleton Tessera owns.

Why: in real use a 2B model wrote good CSS and JavaScript but kept getting the
*boundaries* wrong (a "head" section that emitted a whole empty page, a CSS
section that never closed </style>) and the *names* inconsistent (CSS styled
#city-input while the markup said id="cityInput"). So Tessera takes over both:

  skeleton   <!DOCTYPE>, <head>, <style>…</style>, <body>, <script>…</script>
             and every closing tag are written by Tessera, never by the model
  parts      the model writes only pure body markup, pure JavaScript, pure CSS;
             any wrapper tags it adds anyway are stripped
  order      markup first → its real ids/classes are read with a parser →
             JavaScript is given those exact names → CSS is given the markup's
             names plus the classes the script creates
  checks     each part is validated on its own, then names are cross-checked
             (every #id in CSS / getElementById in JS must exist); a problem
             goes back only to the part that has it
  CSS        cut-off rules are trimmed and duplicate rules removed without a
             model call
  resume     parts that passed are returned so a retry redoes only the rest
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from ..context.continuation import merge_continuation, strip_fences
from ..context.tracker import ContextTracker
from ..core.events import NullEvents
from ..core.interfaces import EventSink, ModelProvider
from ..core.types import Message
from .validators import bracket_problems, html_problems, js_problems, validate_file

PART_ORDER = ("markup", "script", "css")
PART_LABEL = {"markup": "page markup", "script": "JavaScript", "css": "CSS"}

PART_PROMPT = """Stage: filesection.
You write ONE part of a web page. Tessera adds <!DOCTYPE>, <html>, <head>, <style>, <body>, <script> and all
closing tags itself — never write those tags. Output only the raw {label}, no markdown fences, no explanations.
Keep it focused: about {lines} lines."""

GUIDE = {
    "markup": ("Write the contents of <body> only: headings, the input form, the empty-state message and the "
               "container the script fills. Use short lowercase-hyphen ids and classes (e.g. city-input, "
               "add-button, card-grid). No inline styles, no <style>, no <script>."),
    "script": ("Write the JavaScript only. Use exactly the ids and classes listed below — do not invent other ids. "
               "Choose clear class names for elements you create (e.g. weather-card) and use them consistently."),
    "css": ("Write the CSS only. Style exactly the ids and classes listed below, each selector once. Make it "
            "beautiful and responsive as the task asks."),
}
LINES = {"markup": 40, "script": 150, "css": 150}

_WRAPPER_TAGS = re.compile(r"<!DOCTYPE[^>]*>|</?(?:html|head|body|style|script)\b[^>]*>|<meta\b[^>]*>|<title>.*?</title>",
                           re.I | re.S)
_BLOCKS_IN_MARKUP = re.compile(r"<(style|script)\b[^>]*>.*?</\1>", re.I | re.S)
_CSS_SELECTOR_IDS = re.compile(r"#([A-Za-z][\w-]*)")
_CSS_SELECTOR_CLASSES = re.compile(r"\.([A-Za-z][\w-]*)")
_JS_ID_LOOKUPS = re.compile(r"""getElementById\(\s*['"`]([\w-]+)['"`]\s*\)|querySelector(?:All)?\(\s*['"`]#([\w-]+)""")
_JS_CLASSES = re.compile(r"""(?:className\s*=\s*|classList\.(?:add|toggle)\(\s*)['"`]([^'"`]+)['"`]|class\s*=\s*\\?["']([^"'\\$]+)""")


# --------------------------------------------------------------------- helpers
class _NameCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.classes: list[str] = []
        self.title = ""
        self._in_h1 = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.append(a["id"])
        self.classes += (a.get("class") or "").split()
        self._in_h1 = tag == "h1" and not self.title

    def handle_data(self, data):
        if self._in_h1 and data.strip():
            self.title = data.strip()

    def handle_endtag(self, tag):
        if tag == "h1":
            self._in_h1 = False


def markup_names(markup: str) -> tuple[list[str], list[str], str]:
    c = _NameCollector()
    c.feed(markup)
    return list(dict.fromkeys(c.ids)), list(dict.fromkeys(c.classes)), c.title


def script_classes(script: str) -> list[str]:
    out: list[str] = []
    for m in _JS_CLASSES.finditer(script):
        out += (m.group(1) or m.group(2) or "").split()
    return [c for c in dict.fromkeys(out) if re.fullmatch(r"[A-Za-z][\w-]*", c)]


def clean_part(name: str, text: str) -> str:
    t = strip_fences(text.strip("\n"))
    t = re.sub(r"^\s*(?:```\w*|SECTION[^\n]*|WRITE NOW[^\n]*)\n", "", t, flags=re.I)
    if name == "markup":
        t = _BLOCKS_IN_MARKUP.sub("", t)
    t = _WRAPPER_TAGS.sub("", t)
    t = t.replace("```", "")
    return t.strip("\n") + "\n"


def css_rules(css: str) -> list[tuple[str, str]]:
    """Top-level (selector, full rule text) pairs; @media etc. kept whole."""
    rules, depth, start = [], 0, 0
    i = 0
    while i < len(css):
        c = css[i]
        if css.startswith("/*", i):
            end = css.find("*/", i + 2)
            i = len(css) if end < 0 else end + 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                block = css[start : i + 1]
                rules.append((block.split("{", 1)[0].strip(), block.strip()))
                start = i + 1
        i += 1
    return rules


def repair_css(css: str) -> tuple[str, list[str]]:
    """Trim an unfinished trailing rule and drop earlier duplicates of a selector.
    Later duplicates win in CSS anyway, so keeping the last copy preserves behaviour."""
    notes = []
    rules = css_rules(css)
    complete = "\n".join(r for _, r in rules)
    tail = css[sum(len(r) for _, r in rules):] if not rules else css[css.rfind(rules[-1][1]) + len(rules[-1][1]):]
    if tail.strip():
        notes.append("trimmed an unfinished CSS rule at the end")
    seen: dict[str, int] = {}
    for idx, (sel, _) in enumerate(rules):
        if sel and not sel.startswith("@"):
            seen[sel] = idx
    kept = [r for idx, (sel, r) in enumerate(rules) if sel.startswith("@") or not sel or seen.get(sel) == idx]
    if len(kept) < len(rules):
        notes.append(f"removed {len(rules) - len(kept)} duplicate CSS rule(s)")
    return ("\n\n".join(kept) + "\n") if kept else complete, notes


def assemble(parts: dict[str, str], title: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        f"<title>{html.escape(title)}</title>\n<style>\n{parts.get('css', '')}</style>\n</head>\n<body>\n"
        f"{parts.get('markup', '')}<script>\n{parts.get('script', '')}</script>\n</body>\n</html>\n"
    )


def is_cosmetic(problem: str) -> bool:
    """Unused or mismatched CSS selectors make a page look wrong, never break it.
    They get one fix attempt, then are reported as warnings — they never block saving."""
    return problem.startswith("CSS styles ")


def blocking(problems: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {k: [p for p in v if not is_cosmetic(p)] for k, v in problems.items()}
    return {k: v for k, v in out.items() if v}


def check_parts(parts: dict[str, str]) -> dict[str, list[str]]:
    """Problems per part, including cross-part name mismatches."""
    probs: dict[str, list[str]] = {p: [] for p in PART_ORDER}
    markup, script, css = parts.get("markup", ""), parts.get("script", ""), parts.get("css", "")
    probs["markup"] += [p for p in html_problems(f"<body>\n{markup}</body>") if "never closed" in p or "not closed" in p]
    probs["script"] += js_problems(script)
    probs["css"] += bracket_problems(css, "CSS")

    ids, classes, _ = markup_names(markup)
    known_classes = set(classes) | set(script_classes(script))
    js_created_ids = set(re.findall(r"""\.id\s*=\s*['"`]([\w-]+)""", script))
    for m in _JS_ID_LOOKUPS.finditer(script):
        i = m.group(1) or m.group(2)
        if i not in ids and i not in js_created_ids:
            probs["script"].append(f"script looks up id '{i}' but the markup has no such id (ids: {', '.join(ids) or 'none'})")
    for sel, _ in css_rules(css):
        if sel.startswith("@"):
            continue
        for i in _CSS_SELECTOR_IDS.findall(sel):
            if i not in ids and i not in js_created_ids:
                probs["css"].append(f"CSS styles #{i} but the page has no such id (ids: {', '.join(ids) or 'none'})")
        for c in _CSS_SELECTOR_CLASSES.findall(sel):
            if c not in known_classes:
                probs["css"].append(f"CSS styles .{c} but no element uses that class")
    for p in probs:
        probs[p] = list(dict.fromkeys(probs[p]))[:6]
    return {p: v for p, v in probs.items() if v}


# ------------------------------------------------------------------- builder
@dataclass
class PageResult:
    ok: bool
    path: Path
    parts: dict[str, str]
    good_parts: dict[str, str]
    problems: list[str] = field(default_factory=list)
    model_calls: int = 0
    fixes: int = 0
    continuations: int = 0
    notes: list[str] = field(default_factory=list)


class HtmlPageBuilder:
    def __init__(self, model: ModelProvider, tracker: ContextTracker, *, events: EventSink | None = None,
                 part_tokens: int = 1400, max_continuations: int = 3, max_fix_rounds: int = 4,
                 repeat_penalty: float = 1.12) -> None:
        self._model, self._tracker = model, tracker
        self._events = events or NullEvents()
        self._part_tokens, self._max_cont, self._max_fix = part_tokens, max_continuations, max_fix_rounds
        self._repeat_penalty = repeat_penalty
        self.calls = 0

    def build(self, target: Path, task: str, previous: dict[str, str] | None = None,
              extra_check=None) -> PageResult:
        self._extra_check = extra_check
        partial = target.with_name(target.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        parts: dict[str, str] = dict(previous or {})
        res = PageResult(False, target, parts, {})
        if parts:
            self._events.emit("File", f"{target.name}: resuming — keeping {', '.join(PART_LABEL[p] for p in parts)}")
        for name in PART_ORDER:
            if name not in parts:
                parts[name] = self._write(name, task, parts, "", res)
                self._events.emit("File", f"{target.name}: {PART_LABEL[name]} written ({parts[name].count(chr(10))} lines)")
                self._save(partial, parts)

        problems = self._check(parts, res)
        style_fix_used = False
        while res.fixes < self._max_fix:
            hard = blocking(problems)
            if hard:
                target_problems = hard
            elif problems and not style_fix_used:
                target_problems, style_fix_used = problems, True   # one attempt to tidy the styling
            else:
                break
            res.fixes += 1
            name = next(p for p in PART_ORDER if p in target_problems)  # markup first: others depend on it
            self._events.emit("File", f"fix {res.fixes}/{self._max_fix}: {PART_LABEL[name]} — {target_problems[name][0][:90]}")
            parts[name] = self._write(name, task, parts, "\n".join(problems[name]), res)
            self._save(partial, parts)
            problems = self._check(parts, res)

        page = self._save(partial, parts)
        final = validate_file(target.name, page)
        hard = blocking(problems)
        warnings = [p for v in problems.values() for p in v if is_cosmetic(p)]
        if warnings:
            res.notes.append(f"{len(warnings)} styling warning(s), page still works: " + "; ".join(warnings[:3]))
            self._events.emit("File", f"saved with {len(warnings)} styling warning(s) — e.g. {warnings[0][:80]}")
        res.problems = [p for v in hard.values() for p in v] + [p for p in final if p not in str(problems)]
        res.good_parts = {p: t for p, t in parts.items() if p not in hard}
        res.model_calls = self.calls
        if not res.problems:
            os.replace(partial, target)
            res.ok = True
        return res

    # -------------------------------------------------------------- internals
    def _check(self, parts: dict[str, str], res: PageResult) -> dict[str, list[str]]:
        if parts.get("css"):
            fixed, notes = repair_css(parts["css"])
            if notes:
                parts["css"] = fixed
                res.notes += notes
                self._events.emit("File", "CSS auto-repaired: " + "; ".join(notes))
        if parts.get("script"):
            from .js_repair import repair_js

            fixed, notes = repair_js(parts["script"])
            if notes:
                parts["script"] = fixed
                res.notes += notes
                self._events.emit("File", "JavaScript auto-repaired: " + "; ".join(notes))
        problems = check_parts(parts)
        extra = getattr(self, "_extra_check", None)
        if extra:
            found = extra(assemble(parts, "page"))
            if found:   # missing request items (e.g. a "/chat" endpoint) almost always belong in the script
                problems.setdefault("script", []).extend(found)
        return problems

    def _context_for(self, name: str, parts: dict[str, str]) -> str:
        ids, classes, _ = markup_names(parts.get("markup", ""))
        if name == "markup":
            return ""
        lines = [f"IDS IN THE PAGE: {', '.join(ids) or 'none'}", f"CLASSES IN THE PAGE: {', '.join(classes) or 'none'}"]
        if name == "script":
            lines.append("PAGE MARKUP (for reference):\n" + parts.get("markup", "")[:2500])
        if name == "css":
            created = [c for c in script_classes(parts.get("script", "")) if c not in classes]
            lines.append(f"CLASSES THE SCRIPT CREATES: {', '.join(created) or 'none'}")
        return "\n".join(lines)

    def _gen(self, name: str, user: str):
        msgs = [Message.system(PART_PROMPT.format(label=PART_LABEL[name], lines=LINES[name])), Message.user(user)]
        budget = min(self._part_tokens, self._tracker.measure(msgs).safe_output_budget)
        if budget < 64:
            raise RuntimeError("the part prompt leaves no room to write; shorten the description")
        self.calls += 1
        return self._model.generate(msgs, max_output_tokens=budget, temperature=0.3, repeat_penalty=self._repeat_penalty)

    def _write(self, name: str, task: str, parts: dict[str, str], problems: str, res: PageResult) -> str:
        user = f"TASK:\n{task}\n\nWRITE NOW: the {PART_LABEL[name]}. {GUIDE[name]}"
        ctx = self._context_for(name, parts)
        if ctx:
            user += f"\n\n{ctx}"
        if problems:
            user += f"\n\nYOUR PREVIOUS VERSION HAD THESE PROBLEMS — fix all of them:\n{problems}"
        r = self._gen(name, user)
        text, n = r.text, 0
        while r.finish_reason == "length" and n < self._max_cont:
            n += 1
            last = [l for l in text.rstrip().splitlines() if l.strip()][-1:] or [""]
            tail = "\n".join(text.rstrip().splitlines()[-12:])
            r = self._gen(name, f"TASK:\n{task}\n\nYou were writing the {PART_LABEL[name]} and were cut off. "
                                f"It currently ends with:\n{tail}\n\nThe very last line was:\n{last[0]}\n"
                                "Continue EXACTLY after that line. Do not repeat anything already written.")
            text = merge_continuation(text, r.text)
        res.continuations += n
        return clean_part(name, text)

    @staticmethod
    def _save(partial: Path, parts: dict[str, str]) -> str:
        _, _, title = markup_names(parts.get("markup", ""))
        page = assemble(parts, title or partial.name.split(".")[0].replace("_", " ").title())
        partial.write_text(page, encoding="utf-8")
        return page
