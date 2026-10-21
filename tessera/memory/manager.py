"""MemoryManager: the façade the orchestrator sees.

SRP: coordinate the memory components' lifecycle — persist the live window,
archive evicted messages into evidence (+ embeddings + graph links), recall
evidence for a query, store explicit facts, forget. It holds no ranking or
graph logic of its own.

Only real user/assistant messages and explicitly stored facts become
memory. Agent handoffs, router output and tool observations never do.
"""

from __future__ import annotations

import logging
import threading

from ..core.types import Message, Role
from .active_context import ActiveContext
from .associative_graph import AssociativeGraph
from .database import Database
from .embedding_service import EmbeddingService
from .retrieval import RetrievedEvidence, Retriever, format_evidence
from .semantic_cache import SemanticCache
from .terms import extract_terms, python_terms, split_sentences

log = logging.getLogger("tessera.memory")


class MemoryManager:
    def __init__(
        self,
        db: Database,
        embeddings: EmbeddingService,
        graph: AssociativeGraph,
        retriever: Retriever,
        cache: SemanticCache,
    ) -> None:
        self._db = db
        self._emb = embeddings
        self._graph = graph
        self._retriever = retriever
        self._cache = cache
        self._generation = 0
        self._lock = threading.Lock()
        self._msg_ids: dict[int, int] = {}  # id(Message) → messages.id

    # ------------------------------------------------------------- lifecycle
    def attach(self, context: ActiveContext, restore_limit: int) -> None:
        """Restore the persisted window into `context`, then track it."""
        for row in self._db.window_messages(restore_limit):
            msg = Message(Role(row["role"]), row["content"])
            self._msg_ids[id(msg)] = row["id"]
            context.append(msg)  # no hooks yet → no archiving during restore
        context.on_evict(self.archive_message)
        context.on_append(self.persist_message)

    def persist_message(self, message: Message) -> None:
        if id(message) not in self._msg_ids:
            self._msg_ids[id(message)] = self._db.add_message(message.role.value, message.content)

    def archive_message(self, message: Message) -> list[int]:
        msg_id = self._msg_ids.pop(id(message), None)
        if msg_id is not None:
            self._db.mark_archived(msg_id)
        ref = f"message:{msg_id}" if msg_id is not None else "message"
        return self._store(f"{message.role.value}_message", split_sentences(message.content), ref)

    # ------------------------------------------------------------------ API
    def remember(self, text: str, source_ref: str = "explicit") -> list[int]:
        return self._store("fact", split_sentences(text), source_ref)

    def index_source(self, path: str, summary: str, source: str) -> list[int]:
        """Store a source-file summary; graph terms come from `ast`, not the model."""
        ids = self._store("source", [summary], f"file:{path}", extra_terms=python_terms(source))
        return ids

    def recall(self, query: str, k: int = 5) -> list[RetrievedEvidence]:
        return self._retriever.recall(query, k, generation=self._generation)

    def recall_text(self, query: str, k: int = 5) -> str:
        return format_evidence(self.recall(query, k))

    def forget(self, evidence_ids: list[int]) -> None:
        with self._lock:
            self._db.delete_evidence(evidence_ids)
            self._emb.remove(set(evidence_ids))
            self._graph.cache = type(self._graph.cache)()  # links changed
            self._generation += 1

    def stats(self) -> dict[str, int]:
        return {
            "evidence": self._db.evidence_count(),
            "embed_calls": self._emb.embed_calls,
            "embedding_reuses": self._emb.cache_reuses,
            "recall_cache_hits": self._cache.hits,
            "recall_cache_misses": self._cache.misses,
        }

    # --------------------------------------------------------------- private
    def _store(self, kind: str, sentences: list[str], source_ref: str, extra_terms=()) -> list[int]:
        sentences = [s for s in sentences if len(s.strip()) > 2]
        if not sentences:
            return []
        with self._lock:
            ids = [self._db.add_evidence(kind, s, source_ref) for s in sentences]
            self._emb.index_evidence(list(zip(ids, sentences)))
            for eid, s in zip(ids, sentences):
                self._graph.link(eid, extract_terms(s) + list(extra_terms))
            self._generation += 1
        log.debug("stored %d %s evidence rows from %s", len(ids), kind, source_ref)
        return ids
