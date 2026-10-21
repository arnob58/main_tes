"""Console front-end.

Talks to core through the orchestrator / workflow runner / memory APIs and
the event bus; it never imports the model, prompts or storage formats. A
desktop or web UI can replace it without touching core code.

Ctrl+C while a task runs → cancel at the next safe boundary.
Ctrl+C (or Ctrl+D) at the prompt → exit.
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import Callable, TextIO

from .. import __version__
from ..core.events import EventBus
from ..core.orchestrator import Orchestrator, Reply
from ..core.task_state import CancellationToken

COMMANDS = {
    "/help": "list commands",
    "/history": "show the active conversation window",
    "/debug": "toggle per-turn stats",
    "/remember <text>": "store a fact in long-term memory",
    "/recall <query>": "show what memory retrieves for a query",
    "/forget <id …>": "delete memory evidence by id",
    "/skills": "list installed skills and native tools",
    "/workflow <objective>": "plan and run a multi-step, checkpointed workflow",
    "/workflows": "list unfinished workflows",
    "/resume <id>": "continue a paused or interrupted workflow",
    "/approve <id> <step>": "confirm a consequential workflow step, then resume",
    "/stats": "measurements: latency per stage, tokens/s, skill success, memory, VRAM",
    "/quit": "exit",
}


class ConsoleApprover:
    """Asks the person at the keyboard. Thread-safe (tasks run on a worker thread)."""

    def __init__(self, input_fn: Callable[[str], str], out: TextIO) -> None:
        self._input, self._out = input_fn, out
        self._lock = threading.Lock()
        self.always: set[str] = set()

    def approve(self, action: str, detail: str) -> bool:
        if action in self.always:
            return True
        with self._lock:
            print(f"\n  [Approval needed] {action}\n    {detail.replace(chr(10), chr(10) + '    ')}", file=self._out, flush=True)
            try:
                ans = self._input("  Allow? [y]es / [n]o / [a]lways for this kind: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
        if ans in ("a", "always"):
            self.always.add(action)
            return True
        return ans in ("y", "yes")


class ConsoleUI:
    def __init__(
        self,
        orchestrator: Orchestrator,
        events: EventBus,
        *,
        out: TextIO = sys.stdout,
        input_fn: Callable[[str], str] = input,
        system=None,
    ) -> None:
        self._orch = orchestrator
        self._sys = system
        self._out = out
        self._input = input_fn
        self._debug = False
        events.subscribe(self._on_event)

    def _print(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def _on_event(self, channel: str, message: str) -> None:
        self._print(f"  [{channel}] {message}")

    def _in_worker(self, fn):
        """Run fn on a worker thread so Ctrl+C on the main thread can cancel it."""
        token = CancellationToken()
        box: dict = {}

        def target():
            try:
                box["r"] = fn(token)
            except BaseException as exc:  # surface, don't swallow
                box["e"] = exc

        worker = threading.Thread(target=target, daemon=True)
        previous = None
        if threading.current_thread() is threading.main_thread():
            def on_sigint(signum, frame):
                if not token.cancelled:
                    self._print("\n  [Task] Cancelling at the next safe point…")
                token.cancel()
            previous = signal.signal(signal.SIGINT, on_sigint)
        try:
            worker.start()
            while worker.is_alive():
                worker.join(0.1)
        finally:
            if previous is not None:
                signal.signal(signal.SIGINT, previous)
        if "e" in box:
            raise box["e"]
        return box["r"]

    # ------------------------------------------------------------ commands
    def _command(self, line: str) -> bool:
        """Return False to exit."""
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        s = self._sys
        if cmd == "/quit":
            return False
        if cmd == "/history":
            msgs = self._orch.context.messages()
            self._print("  (empty)" if not msgs else "\n".join(f"  {m.role.value}: {m.content[:160]}" for m in msgs))
        elif cmd == "/debug":
            self._debug = not self._debug
            self._print(f"  debug {'on' if self._debug else 'off'}")
        elif cmd in ("/remember", "/recall", "/forget") and (s is None or s.app.memory is None):
            self._print("  Memory is not enabled.")
        elif cmd == "/remember" and arg:
            ids = s.app.memory.remember(arg)
            self._print(f"  Stored as evidence {ids}.")
        elif cmd == "/recall" and arg:
            hits = s.app.memory.recall(arg)
            self._print("  (nothing)" if not hits else "\n".join(f"  #{h.id} {h.score:.2f} {h.text}  ({h.via})" for h in hits))
        elif cmd == "/forget" and arg:
            ids = [int(x) for x in arg.split() if x.isdigit()]
            s.app.memory.forget(ids)
            self._print(f"  Forgot {ids}.")
        elif cmd == "/skills" and s is not None:
            s.engine.registry.refresh()
            for spec in s.engine.registry.all():
                self._print(f"  [{spec.origin}] {spec.signature()}")
            for path, why in s.engine.registry.invalid.items():
                self._print(f"  [invalid] {path}: {why}")
        elif cmd == "/workflow" and arg and s is not None:
            wf = self._in_worker(lambda tok: s.workflows.run(s.workflows.create(arg), tok))
            self._print(wf.summary())
        elif cmd == "/workflows" and s is not None:
            pending = s.workflows.pending()
            self._print("  (none)" if not pending else "\n".join(w.summary() for w in pending))
        elif cmd in ("/resume", "/approve") and s is not None:
            parts = arg.split()
            wf = s.workflows.load(parts[0]) if parts else None
            if wf is None:
                self._print("  Unknown workflow id.")
                return True
            if cmd == "/approve" and len(parts) > 1 and parts[1].isdigit():
                s.workflows.approve(wf, int(parts[1]))
            wf = self._in_worker(lambda tok: s.workflows.run(wf, tok))
            self._print(wf.summary())
        elif cmd == "/stats" and s is not None:
            from ..core.metrics import resource_snapshot

            self._print(s.metrics.report() if s.metrics else "  Metrics disabled.")
            r = resource_snapshot()
            gpu = f"{r.gpu_used_mb:.0f}/{r.gpu_total_mb:.0f} MB" if r.gpu_used_mb is not None else r.note
            self._print(f"  process RSS: {r.rss_mb:.0f} MB · GPU: {gpu}" if r.rss_mb else f"  GPU: {gpu}")
            if s.app.memory is not None:
                self._print(f"  memory: {s.app.memory.stats()}")
            self._print(f"  vision: {'available' if s.vision.supported else getattr(s.vision, 'reason', 'unavailable')}")
        else:
            for k, v in COMMANDS.items():
                self._print(f"  {k:<24} {v}")
        return True

    def loop(self) -> None:
        self._print(f"TESSERA v{__version__}  —  /help for commands, Ctrl+C cancels a running task")
        while True:
            try:
                text = self._input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\nBye.")
                return
            if not text:
                continue
            if text.startswith("/"):
                try:
                    if not self._command(text):
                        self._print("Bye.")
                        return
                except Exception as exc:
                    self._print(f"  [Error] {type(exc).__name__}: {exc}")
                continue
            reply: Reply = self._in_worker(lambda tok: self._orch.handle(text, tok))
            self._print(f"\nTessera:\n{reply.text}")
            if self._debug:
                self._print(
                    f"\n  [stats] {reply.status.value} · route={reply.route.label() if reply.route else '-'}"
                    f" · calls={reply.model_calls} · tokens in/out={reply.input_tokens}/{reply.output_tokens}"
                    f" · {reply.elapsed_s:.1f}s"
                )
