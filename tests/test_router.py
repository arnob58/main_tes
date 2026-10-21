import pytest

from tessera.core.types import Complexity, Message
from tessera.llm.prompts import ROUTER_GRAMMAR
from tessera.llm.router import Router, parse_route
from tessera.llm.scripted import ScriptedModel


@pytest.mark.parametrize(
    "raw, complexity, skills",
    [
        ("Instant|no", Complexity.INSTANT, False),
        ("medium|YES", Complexity.MEDIUM, True),
        ("  Complex | yes \n", Complexity.COMPLEX, True),
        ("Complex|no.", Complexity.COMPLEX, False),
    ],
)
def test_strict_forms(raw, complexity, skills):
    d = parse_route(raw)
    assert (d.complexity, d.needs_skills, d.parsed_cleanly) == (complexity, skills, True)


def test_lenient_recovery_flags_itself():
    d = parse_route("I think this is Complex. Skills: yes")
    assert d.complexity is Complexity.COMPLEX and d.needs_skills and not d.parsed_cleanly


@pytest.mark.parametrize("raw", ["", "banana", "???"])
def test_fallback(raw):
    d = parse_route(raw)
    assert (d.complexity, d.needs_skills, d.parsed_cleanly) == (Complexity.MEDIUM, False, False)


def test_router_uses_grammar_small_budget_and_one_call():
    model = ScriptedModel(queue=["Complex|yes"])
    d = Router(model, max_tokens=8).route("find my resume")
    assert d.label() == "Complex|yes"
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call.grammar == ROUTER_GRAMMAR
    assert call.max_output_tokens == 8 and call.temperature == 0.0


def test_router_truncates_recent_context():
    model = ScriptedModel(queue=["Instant|no"])
    recent = [Message.user("x" * 5000), Message.assistant("y" * 5000)]
    Router(model, context_chars=100).route("again", recent)
    user_msg = model.calls[0].messages[-1].content
    assert len(user_msg) < 400 and "Request: again" in user_msg
