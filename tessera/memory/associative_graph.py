"""Term graph over evidence IDs, with spreading activation.

SRP: maintain term ↔ evidence links and co-occurrence, and spread activation
across them. It never stores sentence text — nodes point at evidence IDs —
and it never decides what is *true*: an association is a lead for the model
to weigh, not a fact.

Activation model (deliberately simple and inspectable)
  * seed terms start at 1.0
  * term → evidence: a term passes activation / sqrt(degree) to each linked
    evidence row (hub terms like "work" spread thin instead of flooding)
  * evidence → term: the row passes its activation × decay to its other terms
  * repeat for `max_hops`; each evidence keeps its best activation and the
    term path that produced it, so every hit is explainable
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .database import Database


@dataclass
class Activation:
    score: float
    path: tuple[str, ...]


@dataclass
class NeighbourhoodCache:
    """LRU of term → evidence ids, invalidated per term on write."""

    capacity: int = 2048
    _data: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, label: str):
        if label in self._data:
            self._data.move_to_end(label)
            self.hits += 1
            return self._data[label]
        self.misses += 1
        return None

    def put(self, label: str, ids: tuple[int, ...]) -> None:
        self._data[label] = ids
        self._data.move_to_end(label)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def invalidate(self, labels: Iterable[str]) -> None:
        for l in labels:
            self._data.pop(l, None)


class AssociativeGraph:
    def __init__(self, db: Database, *, decay: float = 0.6, max_hops: int = 3) -> None:
        self._db = db
        self.decay = decay
        self.max_hops = max_hops
        self.cache = NeighbourhoodCache()
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- writes
    def link(self, evidence_id: int, terms: Iterable[str]) -> None:
        terms = list(dict.fromkeys(t for t in terms if t))
        if not terms:
            return
        with self._db.tx() as c:
            c.executemany("INSERT OR IGNORE INTO nodes(label) VALUES (?)", [(t,) for t in terms])
            marks = ",".join("?" * len(terms))
            ids = [r[0] for r in c.execute(f"SELECT id FROM nodes WHERE label IN ({marks})", terms)]
            c.executemany(
                "INSERT OR IGNORE INTO node_evidence(node_id, evidence_id) VALUES (?, ?)",
                [(n, evidence_id) for n in ids],
            )
            pairs = [(a, b) for a in ids for b in ids if a < b]
            c.executemany(
                "INSERT INTO node_edges(a, b, weight) VALUES (?, ?, 1) "
                "ON CONFLICT(a, b) DO UPDATE SET weight = weight + 1",
                pairs,
            )
        self.cache.invalidate(terms)

    # ------------------------------------------------------------------ reads
    def evidence_for(self, label: str) -> tuple[int, ...]:
        cached = self.cache.get(label)
        if cached is not None:
            return cached
        rows = self._db.query(
            "SELECT ne.evidence_id FROM node_evidence ne JOIN nodes n ON n.id = ne.node_id WHERE n.label = ?",
            (label,),
        )
        ids = tuple(r[0] for r in rows)
        self.cache.put(label, ids)
        return ids

    def terms_for(self, evidence_ids: Iterable[int]) -> dict[int, list[str]]:
        ids = list(evidence_ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._db.query(
            f"SELECT ne.evidence_id AS e, n.label AS l FROM node_evidence ne JOIN nodes n ON n.id = ne.node_id "
            f"WHERE ne.evidence_id IN ({marks})",
            ids,
        )
        out: dict[int, list[str]] = defaultdict(list)
        for r in rows:
            out[r["e"]].append(r["l"])
        return out

    def known_terms(self, candidates: Iterable[str]) -> list[str]:
        cands = list(dict.fromkeys(candidates))
        if not cands:
            return []
        marks = ",".join("?" * len(cands))
        rows = self._db.query(f"SELECT label FROM nodes WHERE label IN ({marks})", cands)
        found = {r[0] for r in rows}
        return [c for c in cands if c in found]

    def related_terms(self, label: str, limit: int = 10) -> list[tuple[str, float]]:
        rows = self._db.query(
            """SELECT n2.label AS l, e.weight AS w FROM nodes n1
               JOIN node_edges e ON e.a = n1.id OR e.b = n1.id
               JOIN nodes n2 ON n2.id = CASE WHEN e.a = n1.id THEN e.b ELSE e.a END
               WHERE n1.label = ? ORDER BY e.weight DESC LIMIT ?""",
            (label, limit),
        )
        return [(r["l"], r["w"]) for r in rows]

    # ------------------------------------------------------------- activation
    def spread(self, seeds: dict[str, float], *, max_evidence: int = 200) -> dict[int, Activation]:
        """Return evidence_id → best activation reached from `seeds`."""
        frontier: dict[str, Activation] = {t: Activation(s, (t,)) for t, s in seeds.items() if s > 0}
        visited_terms: set[str] = set(frontier)
        best: dict[int, Activation] = {}

        for hop in range(self.max_hops):
            if not frontier:
                break
            reached: dict[int, Activation] = {}
            for term, act in frontier.items():
                ids = self.evidence_for(term)
                if not ids:
                    continue
                share = act.score / math.sqrt(len(ids))
                for eid in ids:
                    cur = reached.get(eid)
                    if cur is None or share > cur.score:
                        reached[eid] = Activation(share, act.path)
            for eid, act in reached.items():
                cur = best.get(eid)
                if cur is None or act.score > cur.score:
                    best[eid] = act
            if len(best) >= max_evidence or hop == self.max_hops - 1:
                break
            # evidence → new terms
            next_frontier: dict[str, Activation] = {}
            for eid, terms in self.terms_for(reached).items():
                act = reached[eid]
                for t in terms:
                    if t in visited_terms:
                        continue
                    s = act.score * self.decay
                    cur = next_frontier.get(t)
                    if cur is None or s > cur.score:
                        next_frontier[t] = Activation(s, act.path + (t,))
            # keep the frontier bounded: strongest terms only
            frontier = dict(sorted(next_frontier.items(), key=lambda kv: -kv[1].score)[:32])
            visited_terms.update(frontier)
        return best
