"""Short-term semantic cache: speeds up repeated or near-duplicate recalls.

It stores evidence IDs and scores, never text, and never injects anything
into a prompt by itself. An entry is reused only when a new query embedding
is very close to a cached one *and* long-term memory hasn't changed since
(a generation counter), so a hit can never serve stale results.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CachedRecall:
    query_vec: np.ndarray
    generation: int
    hits: tuple[tuple[int, float, str], ...]  # (evidence_id, score, via)


class SemanticCache:
    def __init__(self, capacity: int = 64, threshold: float = 0.95) -> None:
        self._capacity = capacity
        self._threshold = threshold
        self._entries: OrderedDict[int, CachedRecall] = OrderedDict()
        self._next = 0
        self.hits = 0
        self.misses = 0

    def lookup(self, query_vec: np.ndarray, generation: int) -> CachedRecall | None:
        best_key, best_sim = None, -1.0
        for key, entry in self._entries.items():
            if entry.generation != generation:
                continue
            sim = float(entry.query_vec @ query_vec)
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None and best_sim >= self._threshold:
            self._entries.move_to_end(best_key)
            self.hits += 1
            return self._entries[best_key]
        self.misses += 1
        return None

    def store(self, query_vec: np.ndarray, generation: int, hits) -> None:
        # Drop entries from older generations first; they can never hit again.
        for key in [k for k, e in self._entries.items() if e.generation != generation]:
            del self._entries[key]
        self._entries[self._next] = CachedRecall(query_vec.copy(), generation, tuple(hits))
        self._next += 1
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
