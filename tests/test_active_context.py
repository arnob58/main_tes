import pytest

from tessera.core.types import Message
from tessera.memory.active_context import ActiveContext


def test_fifo_window_and_eviction_order():
    evicted = []
    ctx = ActiveContext(limit=5)
    ctx.on_evict(evicted.append)
    msgs = [Message.user(f"m{i}") for i in range(7)]
    ctx.extend(msgs)
    assert [m.content for m in ctx.messages()] == ["m2", "m3", "m4", "m5", "m6"]
    assert [m.content for m in evicted] == ["m0", "m1"]


def test_system_messages_rejected():
    with pytest.raises(ValueError):
        ActiveContext().append(Message.system("nope"))
