"""Phase 2: persistence, embeddings, graph, cache and associative retrieval.

Uses HashingEmbedder (lexical) because MiniLM weights can't be downloaded in
CI here. The associative test is designed so that the target memory shares
*no words* with the query — only the graph can reach it.
"""

import numpy as np
import pytest

from tessera.app import build_app, build_memory
from tessera.config import Config
from tessera.core.types import Message, Role
from tessera.memory.active_context import ActiveContext
from tessera.memory.database import Database
from tessera.memory.embedding_service import EmbeddingService, HashingEmbedder
from tessera.memory.terms import extract_terms, python_terms, split_sentences

from conftest import standard_model


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        return super().encode(texts)


@pytest.fixture
def mem():
    db = Database(":memory:")
    return build_memory(Config(), db, HashingEmbedder()), db


def test_terms_are_deterministic_and_filtered():
    t = extract_terms("My grandmother baked cookies on Main Street every summer.")
    assert "main street" in t and "grandmother" in t and "cookie" in t
    assert "my" not in t and "every" not in t
    assert python_terms("import os\nclass Foo:\n    def bar(self): pass") == ["os", "foo", "bar"]
    assert python_terms("def broken(:") == []
    assert split_sentences("One thing. Two things! Three?") == ["One thing.", "Two things!", "Three?"]


def test_associative_chain_reaches_unrelated_memory(mem):
    m, _ = mem
    ids = {}
    for key, text in {
        "cookie": "The cookie smell drifted out of the kitchen.",
        "bakery": "That cookie smell was just like the old bakery downtown.",
        "childhood": "The bakery was where I spent much of my childhood.",
        "grandma": "During childhood I visited my grandmother every Sunday.",
        "noise1": "Quarterly taxes are due in April.",
        "noise2": "The printer on the third floor jams constantly.",
    }.items():
        ids[key] = m.remember(text)[0]

    hits = m.recall("cookie smell", k=6)
    by_id = {h.id: h for h in hits}
    assert ids["grandma"] in by_id, "graph traversal should surface the grandmother memory"
    g = by_id[ids["grandma"]]
    assert g.via.startswith("association:") and "childhood" in g.via
    assert g.semantic < 0.2                      # unreachable by similarity alone
    assert ids["noise1"] not in by_id and ids["noise2"] not in by_id
    assert all(h.source_ref == "explicit" for h in hits)  # provenance kept
    order = [h.id for h in hits]
    assert order.index(ids["cookie"]) < order.index(ids["grandma"])  # direct beats distant


def test_graph_stores_ids_not_text(mem):
    m, db = mem
    m.remember("Tessera uses Gemma for responses.")
    cols = {r[1] for r in db.query("PRAGMA table_info(node_evidence)")}
    assert cols == {"node_id", "evidence_id"}
    labels = {r[0] for r in db.query("SELECT label FROM nodes")}
    assert "tessera" in labels and "tessera uses gemma for responses." not in labels


def test_eviction_archives_and_embeds_once():
    db = Database(":memory:")
    emb = CountingEmbedder()
    m = build_memory(Config(), db, emb)
    ctx = ActiveContext(5)
    m.attach(ctx, 5)
    for i in range(6):
        ctx.append(Message.user(f"Message number {i} about llamas."))
    assert db.evidence_count() == 1
    assert emb.texts == ["Message number 0 about llamas."]
    row = db.query("SELECT kind, source_ref FROM evidence")[0]
    assert row["kind"] == "user_message" and row["source_ref"].startswith("message:")
    # identical text later is not re-embedded
    m.remember("Message number 0 about llamas.")
    assert len(emb.texts) == 1 and m.stats()["embedding_reuses"] == 1


def test_window_persists_across_restart(tmp_path):
    path = tmp_path / "db.sqlite"
    cfg = Config(embedder="hashing")
    app = build_app(cfg, standard_model(), db=Database(path))
    app.orchestrator.handle("remember that my cat is Miso")
    app.db.close()

    app2 = build_app(cfg, standard_model(), db=Database(path))
    msgs = app2.orchestrator.context.messages()
    assert [(m.role, m.content) for m in msgs] == [
        (Role.USER, "remember that my cat is Miso"), (Role.ASSISTANT, "final answer")]


def test_index_survives_restart(tmp_path):
    path = tmp_path / "db.sqlite"
    m = build_memory(Config(), Database(path), HashingEmbedder())
    eid = m.remember("The staging server is called Kestrel.")[0]
    m2 = build_memory(Config(), Database(path), HashingEmbedder())
    assert eid in [h.id for h in m2.recall("staging server")]


def test_semantic_cache_hits_and_invalidates(mem):
    m, _ = mem
    m.remember("Rust has a borrow checker.")
    m.recall("borrow checker")
    m.recall("borrow checker")
    assert m.stats()["recall_cache_hits"] == 1
    m.remember("Go has goroutines.")                 # memory changed
    m.recall("borrow checker")
    assert m.stats()["recall_cache_hits"] == 1       # stale entry not reused


def test_forget_removes_everywhere(mem):
    m, db = mem
    eid = m.remember("My locker code is unimportant trivia.")[0]
    m.forget([eid])
    assert db.evidence_count() == 0
    assert m.recall("locker code") == []
    assert db.query("SELECT COUNT(*) FROM node_evidence")[0][0] == 0


def test_orchestrator_injects_memory_but_not_into_evaluate():
    db = Database(":memory:")
    model = standard_model("Complex|no")
    app = build_app(Config(), model, db=db, embedder=HashingEmbedder())
    app.memory.remember("The user's favourite editor is Helix.")
    seen = []
    app.events.subscribe(lambda ch, msg: seen.append((ch, msg)))
    app.orchestrator.handle("which editor should I configure?")
    prompts = {c.messages[0].content.split("\n")[0]: c.messages[-1].content for c in model.calls}
    assert "Helix" in prompts["Stage: interpret."] and "Helix" in prompts["Stage: respond."]
    assert "Helix" not in prompts["Stage: evaluate."]
    assert ("Memory", "1 related memories") in seen


def test_hashing_embedder_shape_and_norm():
    v = HashingEmbedder().encode(["hello world", ""])
    assert v.shape == (2, 384) and v.dtype == np.float32
    assert abs(np.linalg.norm(v[0]) - 1) < 1e-5 and np.linalg.norm(v[1]) == 0


def test_embedding_service_search_order():
    db = Database(":memory:")
    svc = EmbeddingService(HashingEmbedder(), db)
    ids = [db.add_evidence("fact", t) for t in ("apples are red", "bananas are yellow")]
    svc.index_evidence(list(zip(ids, ["apples are red", "bananas are yellow"])))
    hits = svc.search(svc.embed_query("red apples"), 2)
    assert hits[0][0] == ids[0]
