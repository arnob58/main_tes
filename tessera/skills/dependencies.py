"""Missing-dependency detection and (approved) installation.

SRP: turn "No module named 'fitz'" into "pip install pymupdf", ask the
Approver, run the installer, report exactly what happened. Installing
packages changes the user's environment, so it is gated by default.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from ..core.interfaces import Approver

# import name → PyPI distribution, for the common mismatches
PIP_NAMES = {
    "fitz": "pymupdf", "cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml",
    "bs4": "beautifulsoup4", "docx": "python-docx", "pptx": "python-pptx", "sklearn": "scikit-learn",
    "dateutil": "python-dateutil", "dotenv": "python-dotenv", "magic": "python-magic",
    "Crypto": "pycryptodome", "win32api": "pywin32", "win32gui": "pywin32", "serial": "pyserial",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,100}$")


@dataclass(frozen=True)
class InstallOutcome:
    package: str
    installed: bool
    detail: str


Installer = Callable[[str], tuple[bool, str]]


def pip_installer(package: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", package],
        capture_output=True, text=True, timeout=600,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]


class DependencyManager:
    def __init__(self, approver: Approver, installer: Installer = pip_installer, *, auto_approve: bool = False) -> None:
        self._approver = approver
        self._installer = installer
        self._auto = auto_approve
        self._attempted: set[str] = set()

    @staticmethod
    def package_for(module: str) -> str:
        root = module.split(".")[0]
        return PIP_NAMES.get(root, root)

    def ensure(self, module: str) -> InstallOutcome:
        package = self.package_for(module)
        if not _SAFE_NAME.match(package):
            return InstallOutcome(package, False, "refused: suspicious package name")
        if package in self._attempted:
            return InstallOutcome(package, False, "already attempted in this session; not retrying")
        self._attempted.add(package)
        if not self._auto and not self._approver.approve("install_package", f"pip install {package}  (needed for import '{module}')"):
            return InstallOutcome(package, False, "installation not approved")
        ok, log = self._installer(package)
        return InstallOutcome(package, ok, "installed" if ok else f"pip failed: {log[-400:]}")
