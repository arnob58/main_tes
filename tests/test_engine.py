"""Engine logic tested against a fake llama object (no GPU, no model file).

These verify Tessera's own logic — single-instance rule, exact-count
plumbing, budget clamping, grammar caching. They do NOT prove llama.cpp
behaves the same way; that needs the smoke test in README on real hardware.
"""

from dataclasses import dataclass

import pytest

from tessera.config import Config
from tessera.core.types import Message
from tessera.llm.engine import ContextOverflowError, LlamaCppEngine, ModelAlreadyLoadedError

BOS, EOS = 2, 1


@dataclass
class FormatterResponse:
    prompt: str
    stop: list


class FakeLlama:
    metadata = {"tokenizer.chat_template": "{{ messages }}"}

    def __init__(self):
        self.completions = []
        self.resets = 0
        self.closed = False

    def token_bos(self):
        return BOS

    def token_eos(self):
        return EOS

    def detokenize(self, ids, special=False):
        return {BOS: b"<bos>", EOS: b"<eos>"}[ids[0]]

    def tokenize(self, data, add_bos=True, special=False):
        words = data.decode().split()
        return ([BOS] if add_bos else []) + [100 + i for i, _ in enumerate(words)]

    def create_completion(self, **kw):
        self.completions.append(kw)
        return {
            "choices": [{"text": " hello ", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(kw["prompt"]), "completion_tokens": 1},
        }

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def formatter_factory(template, bos, eos):
    assert (bos, eos) == ("<bos>", "<eos>")

    def fmt(messages):
        text = " ".join(f"<{m['role']}> {m['content']}" for m in messages) + " <model>"
        return FormatterResponse(prompt=text, stop=[eos])

    return fmt


compiled = []


def make(cfg=None, fake=None):
    fake = fake or FakeLlama()
    eng = LlamaCppEngine(
        cfg or Config(context_window=50),
        llama_factory=lambda c: fake,
        grammar_compiler=lambda g: compiled.append(g) or f"compiled:{g}",
        formatter_factory=formatter_factory,
    )
    return eng, fake


@pytest.fixture(autouse=True)
def _release():
    yield
    if LlamaCppEngine._live_instance is not None:
        LlamaCppEngine._live_instance.close()
    compiled.clear()


def test_only_one_model_may_be_loaded():
    eng, _ = make()
    with pytest.raises(ModelAlreadyLoadedError):
        make()
    eng.close()
    make()  # allowed again after release


def test_count_matches_tokens_actually_evaluated():
    eng, fake = make()
    msgs = [Message.system("be brief"), Message.user("hi there")]
    n = eng.count_prompt_tokens(msgs)
    r = eng.generate(msgs, max_output_tokens=5)
    sent = fake.completions[0]["prompt"]
    assert sent[0] == BOS and len(sent) == n == r.input_tokens
    assert r.text == "hello" and r.output_tokens == 1 and r.finish_reason == "stop"
    assert "<eos>" in fake.completions[0]["stop"]


def test_output_budget_clamped_to_remaining_context():
    eng, fake = make(Config(context_window=12))
    msgs = [Message.user("a b c d e")]          # 1 bos + 7 words = 8 tokens
    r = eng.generate(msgs, max_output_tokens=500)
    assert fake.completions[0]["max_tokens"] == 4
    assert r.metadata["clamped"] is True


def test_overflow_raises_instead_of_truncating_silently():
    eng, _ = make(Config(context_window=5))
    with pytest.raises(ContextOverflowError):
        eng.generate([Message.user("one two three four five six")])


def test_grammar_compiled_once():
    eng, fake = make()
    for _ in range(3):
        eng.generate([Message.user("x")], grammar="root ::= \"a\"")
    assert compiled == ['root ::= "a"']
    assert fake.completions[-1]["grammar"] == 'compiled:root ::= "a"'


def test_optional_kv_reset():
    eng, fake = make(Config(context_window=50, reset_kv_between_calls=True))
    eng.generate([Message.user("x")])
    eng.generate([Message.user("y")])
    assert fake.resets == 2


def test_missing_chat_template_is_an_error():
    fake = FakeLlama()
    fake.metadata = {}
    with pytest.raises(RuntimeError, match="chat template"):
        make(fake=fake)
    assert LlamaCppEngine._live_instance is None and fake.closed
