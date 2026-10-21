"""Tessera web app — a UI layer, like ui/console.py; talks to core only through the orchestrator.

Endpoints (all JSON unless noted):
  GET  /                  the chat page (static/index.html)
  GET  /api/events        Server-Sent Events: live progress, approval requests, replies
  GET  /api/status        {"busy": bool, "model": str}
  POST /api/chat          {"message": str} → starts a task (one at a time); reply arrives as an event
  POST /api/cancel        stops the running task at the next safe point
  POST /api/approve       {"id": int, "ok": bool} answers an approval request
  POST /api/reset         clears the conversation (not long-term memory)

Only listens on 127.0.0.1 by default: this is a local assistant that can write files.
"""

from __future__ import annotations

import itertools
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"


class EventHub:
    """Fan-out of events to every open browser tab."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, **data) -> None:
        msg = {"type": kind, "time": time.time(), **data}
        with self._lock:
            for q in list(self._subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass


class WebApprover:
    """Asks the browser before risky actions; no answer within `timeout_s` means no."""

    def __init__(self, hub: EventHub, timeout_s: float = 300.0) -> None:
        self._hub, self._timeout = hub, timeout_s
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[threading.Event, list[bool]]] = {}

    def approve(self, action: str, detail: str) -> bool:
        rid = next(self._ids)
        done, answer = threading.Event(), [False]
        self._pending[rid] = (done, answer)
        self._hub.publish("approval", id=rid, action=action, detail=detail[:2000])
        try:
            done.wait(self._timeout)
            return answer[0]
        finally:
            self._pending.pop(rid, None)
            self._hub.publish("approval_closed", id=rid, ok=answer[0])

    def answer(self, rid: int, ok: bool) -> bool:
        item = self._pending.get(rid)
        if item is None:
            return False
        item[1][0] = bool(ok)
        item[0].set()
        return True

    def cancel_all(self) -> None:
        for done, _ in list(self._pending.values()):
            done.set()


class WebSession:
    """Runs one task at a time on a worker thread and reports through the hub."""

    def __init__(self, orchestrator, events, hub: EventHub, approver: WebApprover, model_name: str = "") -> None:
        from ..core.task_state import CancellationToken

        self._orch, self._hub, self._approver = orchestrator, hub, approver
        self._token_cls = CancellationToken
        self._lock = threading.Lock()
        self._cancel = None
        self.model_name = model_name
        events.subscribe(lambda ch, msg: hub.publish("progress", channel=ch, message=msg))

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def start(self, text: str) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        self._cancel = self._token_cls()
        threading.Thread(target=self._run, args=(text, self._cancel), daemon=True).start()
        return True

    def _run(self, text: str, cancel) -> None:
        try:
            self._hub.publish("started", message=text)
            reply = self._orch.handle(text, cancel)
            self._hub.publish("reply", text=reply.text, status=reply.status.value,
                              route=reply.route.label() if reply.route else "",
                              elapsed=round(reply.elapsed_s, 1), model_calls=reply.model_calls,
                              tokens=reply.input_tokens + reply.output_tokens)
        except Exception as exc:   # the orchestrator already catches; this is a last resort
            self._hub.publish("reply", text=f"Something went wrong: {exc}", status="failed")
        finally:
            self._cancel = None
            self._lock.release()
            self._hub.publish("idle")

    def cancel(self) -> bool:
        if self._cancel is None:
            return False
        self._cancel.cancel()
        self._approver.cancel_all()
        return True

    def reset(self) -> bool:
        if self.busy:
            return False
        ctx = self._orch.context
        clear = getattr(ctx, "clear", None)
        if callable(clear):
            clear()
        return True


def make_handler(session: WebSession, hub: EventHub, approver: WebApprover):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet console
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data) -> None:
            self._send(code, json.dumps(data).encode("utf-8"), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise ValueError("request too large")
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def _same_origin(self) -> bool:
            # Block other websites from driving your local assistant through your browser.
            origin = self.headers.get("Origin")
            return origin is None or origin.split("://", 1)[-1] == self.headers.get("Host", "")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(200, {"busy": session.busy, "model": session.model_name})
            if path == "/api/events":
                return self._stream()
            self._send(404, b"Not found", "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.subscribe()
            try:
                self.wfile.write(f"data: {json.dumps({'type': 'hello', 'busy': session.busy, 'model': session.model_name})}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                hub.unsubscribe(q)
                self.close_connection = True

        def do_POST(self):
            if not self._same_origin():
                return self._json(403, {"error": "cross-origin request refused"})
            path = self.path.split("?")[0]
            try:
                data = self._body()
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            if path == "/api/chat":
                text = str(data.get("message", "")).strip()
                if not text:
                    return self._json(400, {"error": "empty message"})
                if not session.start(text):
                    return self._json(409, {"error": "Tessera is still working on the previous message"})
                return self._json(202, {"ok": True})
            if path == "/api/cancel":
                return self._json(200, {"ok": session.cancel()})
            if path == "/api/approve":
                try:
                    rid = int(data.get("id"))
                except (TypeError, ValueError):
                    return self._json(400, {"error": "missing id"})
                return self._json(200, {"ok": approver.answer(rid, bool(data.get("ok")))})
            if path == "/api/reset":
                ok = session.reset()
                return self._json(200 if ok else 409, {"ok": ok})
            self._json(404, {"error": "Not found"})

    return Handler


def serve(orchestrator, events, approver: WebApprover, hub: EventHub, host: str = "127.0.0.1",
          port: int = 8000, model_name: str = "") -> ThreadingHTTPServer:
    session = WebSession(orchestrator, events, hub, approver, model_name)
    server = ThreadingHTTPServer((host, port), make_handler(session, hub, approver))
    server.daemon_threads = True
    return server
