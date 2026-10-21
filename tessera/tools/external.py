"""Talking to other local applications and AI systems.

SRP split:
  discover_chat_apps()  deterministic scan: files/folders matching a name,
                        inspected with ast for an exposed interface
  ChatAdapter           send(message, history) -> reply, one per interface
                        kind: python function, CLI command, HTTP endpoint
  AIConversation        Tessera's side of a bounded, orchestrated dialogue
  tools                 find_chat_app / chat_with_app native tools

Nothing here knows about any particular bot. If no interface can be found
deterministically, the skill engine can still be asked to generate an adapter.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from ..core.events import NullEvents
from ..core.interfaces import Approver, EventSink, ModelProvider
from ..core.types import Message, Observation
from ..skills.contract import SkillSpec

CHAT_FUNCTION_NAMES = {"chat", "respond", "reply", "ask", "answer", "get_response", "generate_response", "send_message", "talk"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages"}


# ------------------------------------------------------------------ discovery
@dataclass
class Candidate:
    path: str
    kind: str                # python_function | cli | http
    detail: dict = field(default_factory=dict)
    score: float = 0.0
    reason: str = ""


def _inspect_python(path: Path) -> list[Candidate]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return []
    found: list[Candidate] = []
    has_input_loop = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
            has_input_loop = True
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.lower() in CHAT_FUNCTION_NAMES:
            n_args = len(node.args.args)
            if 1 <= n_args <= 2:
                found.append(Candidate(str(path), "python_function",
                                       {"file": str(path), "function": node.name, "takes_history": n_args == 2},
                                       0.9, f"defines {node.name}({', '.join(a.arg for a in node.args.args)})"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and dec.args and isinstance(dec.args[0], ast.Constant):
                    route = str(dec.args[0].value)
                    attr = getattr(dec.func, "attr", "")
                    if attr in {"post", "route"} and any(k in route.lower() for k in ("chat", "message", "ask")):
                        found.append(Candidate(str(path), "http", {"route": route, "handler": node.name}, 0.6,
                                               f"HTTP route {route} (start the app, then give its URL)"))
    if has_input_loop and not any(c.kind == "python_function" for c in found):
        found.append(Candidate(str(path), "cli", {"command": [sys.executable, str(path)], "mode": "stdin_line"},
                               0.5, "reads input() in a loop — can be driven as a CLI"))
    return found


def discover_chat_apps(root: str | Path, name: str, *, max_files: int = 4000) -> list[Candidate]:
    root = Path(os.path.expanduser(str(root)))
    needle = "".join(ch for ch in name.lower() if ch.isalnum())
    out: list[Candidate] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        dir_match = needle and needle in "".join(ch for ch in Path(dirpath).name.lower() if ch.isalnum())
        for fn in filenames:
            seen += 1
            if seen > max_files:
                break
            norm = "".join(ch for ch in fn.lower() if ch.isalnum())
            if not (dir_match or (needle and needle in norm)):
                continue
            p = Path(dirpath) / fn
            if fn.endswith(".py"):
                for c in _inspect_python(p):
                    c.score += 0.1 if needle in norm else 0.0
                    out.append(c)
            elif fn.lower().endswith((".exe", ".bat", ".sh", ".cmd")) or (
                os.name != "nt" and "." not in fn and os.access(p, os.X_OK)  # extensionless POSIX executable
            ):
                out.append(Candidate(str(p), "cli", {"command": [str(p)], "mode": "stdin_line"}, 0.3, "executable"))
    return sorted(out, key=lambda c: -c.score)


# ------------------------------------------------------------------- adapters
class ChatAdapter(Protocol):
    def send(self, message: str, history: list[dict]) -> str: ...


_PY_BRIDGE = textwrap.dedent("""
    import importlib.util, json, sys, os
    spec = importlib.util.spec_from_file_location("ext_app", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(sys.argv[1]))
    spec.loader.exec_module(mod)
    req = json.loads(sys.stdin.read())
    fn = getattr(mod, sys.argv[2])
    out = fn(req["message"], req["history"]) if req["takes_history"] else fn(req["message"])
    sys.stdout.write("\\n<<TESSERA_REPLY>>" + json.dumps(str(out)))
""")


class PythonFunctionAdapter:
    """Calls a chat function in a child process (the app's code never runs inside Tessera)."""

    def __init__(self, file: str, function: str, takes_history: bool = False, timeout_s: float = 120) -> None:
        self.file, self.function, self.takes_history, self.timeout = file, function, takes_history, timeout_s

    def send(self, message, history):
        p = subprocess.run([sys.executable, "-c", _PY_BRIDGE, self.file, self.function],
                           input=json.dumps({"message": message, "history": history, "takes_history": self.takes_history}),
                           capture_output=True, text=True, timeout=self.timeout, cwd=str(Path(self.file).parent))
        if "<<TESSERA_REPLY>>" not in p.stdout:
            raise RuntimeError(f"app returned no reply (exit {p.returncode}): {p.stderr[-400:]}")
        return json.loads(p.stdout.rsplit("<<TESSERA_REPLY>>", 1)[1])


class CliAdapter:
    """One process per message: message on stdin (or as an argument), reply = stdout."""

    def __init__(self, command: list[str], mode: str = "stdin_line", timeout_s: float = 120) -> None:
        self.command, self.mode, self.timeout = command, mode, timeout_s

    def send(self, message, history):
        cmd = self.command + ([message] if self.mode == "argv" else [])
        p = subprocess.run(cmd, input=None if self.mode == "argv" else message + "\n", capture_output=True,
                           text=True, timeout=self.timeout)
        text = p.stdout.strip()
        if not text:
            raise RuntimeError(f"no output (exit {p.returncode}): {p.stderr[-300:]}")
        return text


class HttpChatAdapter:
    """POST JSON; `message_field` carries the text, `reply_path` (dot path) locates the answer."""

    def __init__(self, url: str, message_field: str = "message", reply_path: str = "reply",
                 send_history: bool = False, timeout_s: float = 120) -> None:
        self.url, self.field, self.path, self.send_history, self.timeout = url, message_field, reply_path, send_history, timeout_s

    def send(self, message, history):
        body = {self.field: message}
        if self.send_history:
            body["history"] = history
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        for key in self.path.split("."):
            data = data[int(key)] if isinstance(data, list) else data[key]
        return str(data)


def adapter_from(kind: str, detail: dict) -> ChatAdapter:
    if kind == "python_function":
        return PythonFunctionAdapter(detail["file"], detail["function"], bool(detail.get("takes_history")))
    if kind == "cli":
        return CliAdapter(list(detail["command"]), detail.get("mode", "stdin_line"))
    if kind == "http":
        if "url" not in detail:
            raise ValueError("HTTP app found; start it and pass its full URL as detail.url")
        return HttpChatAdapter(detail["url"], detail.get("message_field", "message"), detail.get("reply_path", "reply"))
    raise ValueError(f"unknown adapter kind {kind!r}")


# --------------------------------------------------------------- conversation
CONVERSE_PROMPT = """Stage: converse.
You are talking with another program ("{name}") on the user's behalf.
Conversation goal: {goal}
Write only your next message to it. When the goal is met or the conversation stops being useful, reply with exactly [END]."""


@dataclass
class Turn:
    speaker: str
    text: str


@dataclass
class ConversationResult:
    status: str        # finished | turn_limit | error
    turns: list[Turn]
    error: str = ""


class AIConversation:
    def __init__(self, model: ModelProvider, adapter: ChatAdapter, *, name: str, events: EventSink | None = None,
                 max_turns: int = 6, message_tokens: int = 160) -> None:
        self._model, self._adapter, self._name = model, adapter, name
        self._events = events or NullEvents()
        self._max_turns, self._tokens = max_turns, message_tokens

    def run(self, goal: str, opening: str = "") -> ConversationResult:
        turns: list[Turn] = []
        history: list[dict] = []  # session state, in the shape most chat apps accept
        for i in range(self._max_turns):
            if i == 0 and opening:
                mine = opening
            else:
                transcript = [Message.system(CONVERSE_PROMPT.format(name=self._name, goal=goal))]
                for t in turns[-8:]:
                    transcript.append(Message.assistant(t.text) if t.speaker == "tessera" else Message.user(t.text))
                if not turns:
                    transcript.append(Message.user("(The conversation has not started. Write your first message.)"))
                mine = self._model.generate(transcript, max_output_tokens=self._tokens, temperature=0.5).text.strip()
            if not mine or mine.strip() == "[END]":
                return ConversationResult("finished", turns)
            turns.append(Turn("tessera", mine))
            self._events.emit("Conversation", f"→ {self._name}: {mine[:80]}")
            try:
                reply = self._adapter.send(mine, list(history))
            except Exception as exc:
                return ConversationResult("error", turns, f"{type(exc).__name__}: {exc}")
            history += [{"role": "user", "content": mine}, {"role": "assistant", "content": reply}]
            turns.append(Turn(self._name, reply))
            self._events.emit("Conversation", f"← {self._name}: {reply[:80]}")
        return ConversationResult("turn_limit", turns)


# ---------------------------------------------------------------------- tools
class FindChatAppTool:
    spec = SkillSpec(
        name="find_chat_app",
        description="Find a local chatbot/app by name and work out how to talk to it (python function, CLI or HTTP)",
        parameters={"name": "str - e.g. 'GPT Luna'", "root": "str - folder to search, default ~/Desktop"},
        origin="native",
    )

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        root = str(arguments.get("root") or "~/Desktop")
        if not Path(os.path.expanduser(root)).exists():
            return Observation("error", self.spec.name, stderr=f"folder not found: {root}")
        cands = discover_chat_apps(root, str(arguments.get("name", "")))
        return Observation("success", self.spec.name,
                           result=[asdict(c) for c in cands[:5]] or "no candidates found",
                           notes=("pass the chosen candidate's kind and detail to chat_with_app",))


class ChatWithAppTool:
    spec = SkillSpec(
        name="chat_with_app",
        description="Hold a bounded conversation with another local app/AI through a discovered interface",
        parameters={"kind": "str - python_function | cli | http", "detail": "dict - from find_chat_app",
                    "goal": "str - what the conversation should achieve", "name": "str - the app's name",
                    "max_turns": "int - default 6", "opening": "str - optional first message"},
        origin="native",
    )

    def __init__(self, model: ModelProvider, approver: Approver, events: EventSink | None = None) -> None:
        self._model, self._approver, self._events = model, approver, events

    def invoke(self, arguments, *, task_id, purpose) -> Observation:
        try:
            adapter = adapter_from(str(arguments.get("kind")), dict(arguments.get("detail") or {}))
        except (ValueError, KeyError) as exc:
            return Observation("error", self.spec.name, stderr=str(exc))
        name = str(arguments.get("name") or "the app")
        if not self._approver.approve("run_external_app", f"start and talk to {name} via {arguments.get('kind')}"):
            return Observation("denied", self.spec.name, result="running the external app was not approved")
        convo = AIConversation(self._model, adapter, name=name, events=self._events,
                               max_turns=int(arguments.get("max_turns", 6)))
        res = convo.run(str(arguments.get("goal") or purpose), str(arguments.get("opening", "")))
        transcript = "\n".join(f"{t.speaker}: {t.text}" for t in res.turns)
        return Observation("success" if res.status != "error" else "error", self.spec.name,
                           result={"status": res.status, "turns": len(res.turns)}, stdout=transcript, stderr=res.error)
