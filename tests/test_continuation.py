"""The cut-off reported from real use: a reply longer than the stage's output cap."""

from tessera.agents.agent import CONTINUE_INSTRUCTION
from tessera.app import build_app
from tessera.config import Config
from tessera.llm.router import parse_route
from tessera.llm.scripted import ScriptedModel

ANSWER = " ".join(f"word{i}" for i in range(1200))  # needs 1200 "tokens"


def continuing_respond(messages):
    """Return the rest of ANSWER, using only what the prompt shows was already said."""
    last = messages[-1].content
    if CONTINUE_INSTRUCTION not in last:
        return ANSWER
    if "YOUR ANSWER SO FAR ENDS WITH:" in last:            # fresh-context path
        said = last.split("YOUR ANSWER SO FAR ENDS WITH:\n", 1)[1].split("\n\n", 1)[0]
    else:                                                  # appended-assistant path
        said = messages[-2].content
    tail = said.split()[-1]
    idx = ANSWER.split().index(tail)
    return " ".join(ANSWER.split()[idx + 1:])


def model(window=8192):
    return (ScriptedModel(context_window=window)
            .when_agent("router", "Instant|no")
            .when_agent("interpret", "INTENT: long answer")
            .when_agent("respond", continuing_respond))


def test_reply_longer_than_cap_comes_back_complete():
    m = model()
    reply = build_app(Config(respond_max_tokens=512, max_continuations=3), m).orchestrator.handle("tell me everything")
    assert reply.text == ANSWER                             # nothing lost, nothing duplicated
    respond_calls = [c for c in m.calls if c.messages[0].content.startswith("Stage: respond.")]
    assert len(respond_calls) == 3                          # 512 + 512 + 176


def test_fresh_context_continuation_when_window_is_tight():
    m = model(window=1400)
    reply = build_app(Config(respond_max_tokens=500, max_continuations=4), m).orchestrator.handle("tell me everything")
    assert reply.text == ANSWER
    prompts = [c.messages[-1].content for c in m.calls if c.messages[0].content.startswith("Stage: respond.")]
    assert any("YOUR ANSWER SO FAR ENDS WITH" in p for p in prompts)


def test_still_unfinished_is_said_plainly():
    m = model()
    reply = build_app(Config(respond_max_tokens=200, max_continuations=1), m).orchestrator.handle("tell me everything")
    assert reply.text.endswith("[This answer was cut short at the length limit. Ask me to continue.]")
    assert reply.text.startswith("word0 word1")


def test_continuations_can_be_disabled():
    m = model()
    reply = build_app(Config(respond_max_tokens=100, max_continuations=0), m).orchestrator.handle("x")
    assert len(reply.text.split()) == 100


def test_router_prompt_routes_build_requests():
    from tessera.llm.prompts import ROUTER_PROMPT
    assert "create me a website tracker -> Complex|yes" in ROUTER_PROMPT
    assert parse_route("Complex|yes").needs_skills
