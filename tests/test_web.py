"""Web app: real HTTP round trip with a scripted model, approvals and cross-origin refusal."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from tessera.app import build_app
from tessera.config import Config
from tessera.web.server import EventHub, WebApprover, serve

from conftest import standard_model


@pytest.fixture
def web():
    hub = EventHub()
    approver = WebApprover(hub, timeout_s=5)
    app = build_app(Config(), standard_model("Instant|no"))
    server = serve(app.orchestrator, app.events, approver, hub, port=0, model_name="test")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, hub, approver, app
    server.shutdown()


def post(url, body, headers=None):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(req, timeout=5)


def test_page_is_served(web):
    base, *_ = web
    html = urllib.request.urlopen(base + "/").read().decode()
    assert "<title>Tessera</title>" in html and "/api/events" in html


def test_chat_round_trip_publishes_progress_and_reply(web):
    base, hub, _, _ = web
    q = hub.subscribe()
    assert post(base + "/api/chat", {"message": "hello"}).status == 202
    seen, deadline = [], time.time() + 10
    while time.time() < deadline:
        m = q.get(timeout=10)
        seen.append(m)
        if m["type"] == "idle":
            break
    kinds = [m["type"] for m in seen]
    assert "progress" in kinds and "reply" in kinds and kinds[-1] == "idle"
    reply = next(m for m in seen if m["type"] == "reply")
    assert reply["text"] and reply["status"] == "completed"


def test_empty_message_rejected(web):
    base, *_ = web
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base + "/api/chat", {"message": "  "})
    assert e.value.code == 400


def test_other_websites_cannot_drive_tessera(web):
    base, *_ = web
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base + "/api/chat", {"message": "hi"}, {"Origin": "https://evil.example"})
    assert e.value.code == 403


def test_browser_approval_round_trip(web):
    base, hub, approver, _ = web
    q = hub.subscribe()
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ok", approver.approve("overwrite_file", "x.html")))
    t.start()
    req = q.get(timeout=5)
    assert req["type"] == "approval" and req["action"] == "overwrite_file"
    post(base + "/api/approve", {"id": req["id"], "ok": True})
    t.join(5)
    assert result["ok"] is True


def test_reset_clears_conversation_only(web):
    base, _, _, app = web
    app.orchestrator.handle("remember this turn")
    assert len(app.orchestrator.context) > 0
    post(base + "/api/reset", {})
    assert len(app.orchestrator.context) == 0
