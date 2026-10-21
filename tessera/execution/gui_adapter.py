"""GUI control behind a small protocol; PyAutoGUI is one implementation.

SRP: move the mouse, type, take screenshots, launch apps. No decisions.

Coordinates in Tessera are *normalised* to a 0–1000 grid on both axes and
converted to pixels at action time from the current screen size, so the same
action works at 1366×768 and 3840×2160.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

GRID = 1000


@dataclass(frozen=True)
class ScreenSize:
    width: int
    height: int

    def to_pixels(self, x: float, y: float) -> tuple[int, int]:
        if not (0 <= x <= GRID and 0 <= y <= GRID):
            raise ValueError(f"coordinates ({x}, {y}) outside the 0–{GRID} grid")
        return round(x / GRID * (self.width - 1)), round(y / GRID * (self.height - 1))


@runtime_checkable
class GuiDriver(Protocol):
    def available(self) -> tuple[bool, str]: ...
    def screen_size(self) -> ScreenSize: ...
    def screenshot(self, path: Path) -> ScreenSize: ...
    def click(self, px: int, py: int, *, button: str = "left", clicks: int = 1) -> None: ...
    def type_text(self, text: str) -> None: ...
    def press(self, key: str) -> None: ...
    def hotkey(self, *keys: str) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def open_url(self, url: str, browser: str = "") -> None: ...
    def launch(self, app: str) -> None: ...


class PyAutoGuiDriver:
    def __init__(self, pause_s: float = 0.25) -> None:
        self._pause = pause_s
        self._pg = None

    def _gui(self):
        if self._pg is None:
            import pyautogui  # deferred: needs a display at import time on Linux

            pyautogui.FAILSAFE = True   # slam the mouse into a corner to abort
            pyautogui.PAUSE = self._pause
            self._pg = pyautogui
        return self._pg

    def available(self) -> tuple[bool, str]:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False, "no graphical display (DISPLAY/WAYLAND_DISPLAY unset)"
        try:
            self._gui()
        except Exception as exc:
            return False, f"pyautogui unavailable: {type(exc).__name__}: {exc}"
        return True, "ok"

    def screen_size(self) -> ScreenSize:
        w, h = self._gui().size()
        return ScreenSize(int(w), int(h))

    def screenshot(self, path: Path) -> ScreenSize:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = self._gui().screenshot()
        img.save(path)
        # Screenshots on HiDPI displays can be larger than the logical screen;
        # actions use logical size, so report that.
        return self.screen_size()

    def click(self, px, py, *, button="left", clicks=1):
        self._gui().click(px, py, button=button, clicks=clicks)

    def type_text(self, text):
        self._gui().write(text, interval=0.02)

    def press(self, key):
        self._gui().press(key)

    def hotkey(self, *keys):
        self._gui().hotkey(*keys)

    def scroll(self, amount):
        self._gui().scroll(int(amount))

    def open_url(self, url, browser=""):
        if browser.lower() in ("chrome", "google chrome"):
            exe = _chrome_path()
            if exe:
                subprocess.Popen([exe, url])
                return
        webbrowser.open(url)

    def launch(self, app):
        system = platform.system()
        if system == "Windows":
            os.startfile(app)  # type: ignore[attr-defined]  # noqa: S606
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", app])
        else:
            subprocess.Popen([shutil.which(app) or app])


def _chrome_path() -> str | None:
    candidates = [
        shutil.which("google-chrome"), shutil.which("chrome"), shutil.which("chromium"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)
