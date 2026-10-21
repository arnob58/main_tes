"""Desktop tools: look at the screen; run an observe-act loop toward a goal."""

from __future__ import annotations

from ..core.types import Observation
from ..execution.artifacts import WorkspaceManager
from ..execution.desktop_loop import DesktopLoop
from ..execution.gui_adapter import GuiDriver
from ..llm.vision import VisionObserver
from ..skills.contract import SkillSpec


class LookAtScreenTool:
    spec = SkillSpec(
        name="look_at_screen",
        description="Take a screenshot and describe what is visible (requires a vision-capable model)",
        parameters={"question": "str - what to look for, default: describe the screen"},
        origin="native",
    )

    def __init__(self, driver: GuiDriver, vision: VisionObserver, workspaces: WorkspaceManager) -> None:
        self._driver, self._vision, self._ws = driver, vision, workspaces

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        ok, why = self._driver.available()
        if not ok:
            return Observation("unavailable", self.spec.name, result=f"GUI unavailable: {why}")
        shot = self._ws.for_task(task_id) / "screens" / "look.png"
        self._driver.screenshot(shot)
        seen = self._vision.observe(shot, str(arguments.get("question") or "Describe this screen concisely."))
        status = {"ok": "success", "unsupported": "unavailable"}.get(seen.status, "error")
        return Observation(status, self.spec.name, result=seen.text, artifacts=(str(shot),))


class DesktopTaskTool:
    spec = SkillSpec(
        name="desktop_task",
        description="Operate desktop apps/browser with mouse and keyboard, watching the screen, until a goal is reached",
        parameters={
            "goal": "str - what should be true at the end",
            "start_url": "str - optional page to open first",
            "browser": "str - optional, e.g. chrome",
            "max_steps": "int - default 15",
        },
        origin="native",
    )

    def __init__(self, loop: DesktopLoop, workspaces: WorkspaceManager) -> None:
        self._loop, self._ws = loop, workspaces

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        goal = str(arguments.get("goal") or purpose)
        res = self._loop.run(
            goal, self._ws.for_task(task_id),
            start_url=str(arguments.get("start_url", "")), browser=str(arguments.get("browser", "")),
            max_steps=int(arguments.get("max_steps", 0)) or None,
        )
        status = {"done": "success", "unsupported": "unavailable", "needs_user": "waiting_for_user",
                  "denied": "denied"}.get(res.status, "error")
        trail = [f"{s.step}. {s.action.get('type', '?')} → {s.outcome}" for s in res.steps[-10:]]
        return Observation(
            status, self.spec.name,
            result={"status": res.status, "summary": res.summary, "steps": len(res.steps)},
            stdout="\n".join(trail),
            artifacts=tuple(s.screenshot for s in res.steps if s.screenshot)[-3:],
            notes=(f"desktop loop ended: {res.status}",),
        )
