"""Hybrid retrieval: semantic search + keywords + spreading activation + recency.

SRP: rank evidence for a query. It reads from EmbeddingService and
AssociativeGraph; it writes nothing except access statistics.

Every result says *how* it was reached ("semantic", "keyword", or an
association path such as cookie → bakery → childhood) so the model can
judge how much weight it deserves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .associative_graph import AssociativeGraph
from .database import Database
from .embedding_service import EmbeddingService
from .semantic_cache import SemanticCache
from .terms import extract_terms


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.55
    graph: float = 0.9
    keyword: float = 0.25
    recency: float = 0.05
    recency_half_life_days: float = 30.0
    semantic_seed_k: int = 5
    semantic_seed_min: float = 0.35
    min_score: float = 0.12


@dataclass(frozen=True)
class RetrievedEvidence:
    id: int
    text: str
    score: float
    via: str
    source_ref: str
    semantic: float


class Retriever:
    def __init__(
        self,
        db: Database,
        embeddings: EmbeddingService,
        graph: AssociativeGraph,
        cache: SemanticCache,
        weights: RetrievalWeights = RetrievalWeights(),
    ) -> None:
        self._db = db
        self._emb = embeddings
        self._graph = graph
        self._cache = cache
        self.w = weights

    def recall(self, query: str, k: int = 5, *, generation: int = 0) -> list[RetrievedEvidence]:
        if not query.strip():
            return []
        qvec = self._emb.embed_query(query)

        cached = self._cache.lookup(qvec, generation)
        if cached is not None:
            scored = list(cached.hits)
        else:
            scored = self._rank(query, qvec)
            self._cache.store(qvec, generation, scored)

        top = scored[:k]
        rows = self._db.get_evidence(eid for eid, _, _ in top)
        sims = self._emb.similarity(qvec, [eid for eid, _, _ in top])
        self._db.touch_evidence(rows)
        out = []
        for eid, score, via in top:
            row = rows.get(eid)
            if row is None:  # deleted since caching (generation should prevent this)
                continue
            out.append(
                RetrievedEvidence(eid, row.text, round(score, 4), via, row.source_ref, round(sims.get(eid, 0.0), 4))
            )
        return out

    def _rank(self, query: str, qvec) -> list[tuple[int, float, str]]:
        w = self.w
        q_terms = extract_terms(query)
        known = self._graph.known_terms(q_terms)

        semantic_hits = self._emb.search(qvec, max(w.semantic_seed_k * 4, 20))
        sem = dict(semantic_hits)

        # Seeds: exact query terms at full strength, plus the terms of strong
        # semantic hits (scaled by similarity) — this is how "cookie smell"
        # can reach "bakery" even if the word "cookie" never matched exactly.
        seeds: dict[str, float] = {t: 1.0 for t in known}
        strong = [(eid, s) for eid, s in semantic_hits[: w.semantic_seed_k] if s >= w.semantic_seed_min]
        for eid, terms in self._graph.terms_for(e for e, _ in strong).items():
            s = sem[eid]
            for t in terms:
                seeds[t] = max(seeds.get(t, 0.0), 0.8 * s)

        activation = self._graph.spread(seeds)
        candidates = set(sem) | set(activation)
        rows = self._db.get_evidence(candidates)
        missing_sem = [c for c in candidates if c not in sem]
        sem.update(self._emb.similarity(qvec, missing_sem))
        term_map = self._graph.terms_for(candidates)
        q_set = set(q_terms)
        now = time.time()

        results = []
        for eid in candidates:
            row = rows.get(eid)
            if row is None:
                continue
            s = max(sem.get(eid, 0.0), 0.0)
            act = activation.get(eid)
            g = act.score if act else 0.0
            overlap = len(q_set & set(term_map.get(eid, ()))) / max(len(q_set), 1)
            age_days = max(now - row.created_at, 0) / 86400
            recency = math.exp(-age_days * math.log(2) / w.recency_half_life_days)
            score = w.semantic * s + w.graph * g + w.keyword * overlap + w.recency * recency
            if score < w.min_score:
                continue
            if act and len(act.path) > 1 and w.graph * g >= w.semantic * s:
                via = "association: " + " → ".join(act.path)
            elif overlap > 0 and w.keyword * overlap >= w.semantic * s:
                via = "keyword"
            else:
                via = "semantic"
            results.append((eid, score, via))
        results.sort(key=lambda r: -r[1])
        return results


def format_evidence(items: list[RetrievedEvidence]) -> str:
    """Prompt block. Labelled as leads, not facts."""
    if not items:
        return ""
    lines = [f"[#{e.id}] {e.text}  ({e.via})" for e in items]
    return "Possibly relevant memories (associations may be loose; verify before relying on them):\n" + "\n".join(lines)
