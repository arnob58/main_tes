"""Observation → action loop for desktop tasks.

SRP: run screenshot → (multimodal model picks an action) → act → screenshot
until the goal is done, the step budget runs out, or the user must decide.

* One multimodal call per step both describes the screen and chooses the
  next action, as strict JSON — cheaper than separate observe/decide calls.
* Every action is tied to the screenshot it was chosen from; if the screen
  size changed since then, the action is not executed blindly — a fresh
  screenshot is taken instead.
* Predictable sequences ("macros") run without consulting the model between
  steps; control returns to the model afterwards.
* Consequential actions (submit, send, buy, apply…) and all text typing into
  the GUI pass through the Approver. Nothing is reported as done unless the
  model observed it on screen.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.events import NullEvents
from ..core.interfaces import Approver, EventSink
from ..llm.vision import VisionObserver
from .gui_adapter import GRID, GuiDriver, ScreenSize

STEP_PROMPT = """You control a computer to reach a GOAL. Look at the screenshot.
Reply with ONE JSON object, nothing else:
{{"observation": "<what is on screen that matters>", "action": {{"type": "<type>", ...}}}}
Action types:
 click {{"x": 0-{grid}, "y": 0-{grid}, "target": "<what you click>", "consequential": false}}
 double_click {{"x", "y", "target"}}
 type {{"text": "..."}}            press {{"key": "enter"}}      hotkey {{"keys": ["ctrl", "l"]}}
 scroll {{"amount": -5}}           open_url {{"url": "https://..."}}   wait {{"seconds": 2}}
 done {{"summary": "<what was achieved, as seen on screen>"}}
 ask_user {{"question": "..."}}     fail {{"reason": "..."}}
Coordinates are on a {grid}×{grid} grid over the whole screenshot (0,0 = top-left).
Set "consequential": true for anything that submits, sends, buys, applies, deletes or publishes.
Only say done when the screenshot shows the goal is achieved.

GOAL: {goal}
STEP: {step} of {max_steps}
RECENT ACTIONS:
{history}"""

CONSEQUENTIAL_WORDS = re.compile(r"\b(submit|send|apply|purchase|buy|pay|delete|publish|post|confirm|place order)\b", re.I)
_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class StepRecord:
    step: int
    screenshot: str
    screen: tuple[int, int]
    observation: str
    action: dict[str, Any]
    outcome: str


@dataclass
class DesktopResult:
    status: str                     # done | failed | needs_user | step_limit | unsupported | denied
    summary: str
    steps: list[StepRecord] = field(default_factory=list)


def parse_step(text: str) -> tuple[str, dict[str, Any]]:
    m = _JSON.search(text)
    if not m:
        raise ValueError("no JSON object in model reply")
    data = json.loads(m.group(0))
    action = data.get("action")
    if not isinstance(action, dict) or "type" not in action:
        raise ValueError("reply lacks action.type")
    return str(data.get("observation", "")), action


class DesktopLoop:
    def __init__(self, driver: GuiDriver, vision: VisionObserver, approver: Approver, *,
                 events: EventSink | None = None, max_steps: int = 15, settle_s: float = 0.8,
                 step_tokens: int = 220) -> None:
        self._driver = driver
        self._vision = vision
        self._approver = approver
        self._events = events or NullEvents()
        self._max_steps = max_steps
        self._settle = settle_s
        self._step_tokens = step_tokens

    def run(self, goal: str, workdir: Path, *, start_url: str = "", browser: str = "",
            macro: list[dict] | None = None, max_steps: int | None = None) -> DesktopResult:
        ok, why = self._driver.available()
        if not ok:
            return DesktopResult("unsupported", f"GUI control unavailable: {why}")
        if not self._vision.supported:
            return DesktopResult("unsupported", f"Cannot see the screen: {getattr(self._vision, 'reason', 'vision unsupported')}")

        shots = workdir / "screens"
        steps: list[StepRecord] = []
        limit = max_steps or self._max_steps
        if start_url:
            self._driver.open_url(start_url, browser)
            self._sleep()
        for action in macro or []:  # predictable steps: no model call between them
            outcome = self._act(action, self._driver.screen_size(), self._driver.screen_size())
            steps.append(StepRecord(len(steps) + 1, "", (0, 0), "(macro)", action, outcome))
            if outcome.startswith(("denied", "error")):
                return DesktopResult("denied" if outcome.startswith("denied") else "failed", outcome, steps)

        parse_failures = 0
        for step in range(1, limit + 1):
            shot = shots / f"step_{step:03d}.png"
            size = self._driver.screenshot(shot)
            history = "\n".join(f"{s.step}. {json.dumps(s.action)} → {s.outcome}" for s in steps[-6:]) or "(none)"
            prompt = STEP_PROMPT.format(grid=GRID, goal=goal, step=step, max_steps=limit, history=history)
            seen = self._vision.observe(shot, prompt, max_output_tokens=self._step_tokens)
            if seen.status != "ok":
                return DesktopResult("unsupported" if seen.status == "unsupported" else "failed", seen.text, steps)
            try:
                observation, action = parse_step(seen.text)
                parse_failures = 0
            except (ValueError, json.JSONDecodeError) as exc:
                parse_failures += 1
                steps.append(StepRecord(step, str(shot), (size.width, size.height), seen.text[:200], {}, f"unparsed: {exc}"))
                if parse_failures >= 2:
                    return DesktopResult("failed", "model replies were not valid actions twice in a row", steps)
                continue

            kind = action.get("type")
            self._events.emit("Desktop", f"step {step}: {kind} — {observation[:80]}")
            if kind == "done":
                steps.append(StepRecord(step, str(shot), (size.width, size.height), observation, action, "done"))
                return DesktopResult("done", str(action.get("summary") or observation), steps)
            if kind == "ask_user":
                steps.append(StepRecord(step, str(shot), (size.width, size.height), observation, action, "needs user"))
                return DesktopResult("needs_user", str(action.get("question", "")), steps)
            if kind == "fail":
                steps.append(StepRecord(step, str(shot), (size.width, size.height), observation, action, "gave up"))
                return DesktopResult("failed", str(action.get("reason", "")), steps)

            outcome = self._act(action, size, self._driver.screen_size())
            steps.append(StepRecord(step, str(shot), (size.width, size.height), observation, action, outcome))
            if outcome.startswith("denied"):
                return DesktopResult("denied", outcome, steps)
            self._sleep()
        return DesktopResult("step_limit", f"stopped after {limit} steps without confirming the goal", steps)

    # ------------------------------------------------------------- actions
    def _act(self, action: dict, shot_size: ScreenSize, now: ScreenSize) -> str:
        kind = action.get("type")
        d = self._driver
        try:
            if kind in ("click", "double_click"):
                if shot_size != now:
                    return "skipped: screen size changed since the screenshot; re-observing"
                px, py = shot_size.to_pixels(float(action["x"]), float(action["y"]))
                target = str(action.get("target", ""))
                if action.get("consequential") or CONSEQUENTIAL_WORDS.search(target):
                    if not self._approver.approve("gui_consequential", f"click '{target}' at ({px},{py})"):
                        return f"denied: user did not approve clicking '{target}'"
                d.click(px, py, clicks=2 if kind == "double_click" else 1)
                return f"clicked {target or (px, py)}"
            if kind == "type":
                text = str(action.get("text", ""))
                if not self._approver.approve("gui_type", f"type {text[:80]!r}"):
                    return "denied: typing not approved"
                d.type_text(text)
                return f"typed {len(text)} chars"
            if kind == "press":
                key = str(action.get("key", ""))
                if key.lower() == "enter" and action.get("consequential"):
                    if not self._approver.approve("gui_consequential", "press enter to submit"):
                        return "denied: submission not approved"
                d.press(key)
                return f"pressed {key}"
            if kind == "hotkey":
                d.hotkey(*[str(k) for k in action.get("keys", [])])
                return f"hotkey {'+'.join(action.get('keys', []))}"
            if kind == "scroll":
                d.scroll(int(action.get("amount", 0)))
                return f"scrolled {action.get('amount')}"
            if kind == "open_url":
                d.open_url(str(action["url"]), str(action.get("browser", "")))
                return f"opened {action['url']}"
            if kind == "launch":
                d.launch(str(action["app"]))
                return f"launched {action['app']}"
            if kind == "wait":
                time.sleep(min(float(action.get("seconds", 1)), 10))
                return "waited"
            return f"error: unknown action type {kind!r}"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def _sleep(self) -> None:
        if self._settle:
            time.sleep(self._settle)
