"""Embeddings: one loaded MiniLM, cached vectors, a NumPy index.

SRP split:
  Embedder        — protocol: texts → (n, dim) L2-normalised float32
  MiniLMEmbedder  — sentence-transformers on CPU, loaded lazily, once
  HashingEmbedder — deterministic, dependency-free stand-in for tests and
                    for running without the model; it captures lexical
                    overlap only and says so in its name
  EmbeddingService— embeds through the DB cache so identical text is never
                    re-embedded, and keeps an in-memory matrix for search
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Protocol, Sequence

import numpy as np

from .database import Database


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    dim = 384

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2") -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # heavy import, deferred

                self._model = SentenceTransformer(self.name, device="cpu")
                dim_fn = getattr(self._model, "get_embedding_dimension", None) or \
                    self._model.get_sentence_embedding_dimension  # renamed in newer sentence-transformers
                got = dim_fn()
                if got != self.dim:
                    raise RuntimeError(f"{self.name} produces {got}-d vectors, expected {self.dim}")
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._load().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        return vecs.astype(np.float32, copy=False)


_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Feature-hashed word + character-trigram vectors. Lexical, not semantic."""

    name = "hashing-384"
    dim = 384

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            feats = [w.rstrip("s") for w in words]
            for w in words:
                padded = f"#{w}#"
                feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


def text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


class EmbeddingService:
    def __init__(self, embedder: Embedder, db: Database) -> None:
        self._embedder = embedder
        self._db = db
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self.embed_calls = 0
        self.cache_reuses = 0
        self._load_index()

    @property
    def model_name(self) -> str:
        return self._embedder.name

    def _load_index(self) -> None:
        rows = self._db.all_embeddings(self._embedder.name)
        if rows:
            self._ids = [r["evidence_id"] for r in rows]
            self._matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])

    def embed_query(self, text: str) -> np.ndarray:
        self.embed_calls += 1
        return self._embedder.encode([text])[0]

    def index_evidence(self, items: Sequence[tuple[int, str]]) -> None:
        """Embed and store vectors for (evidence_id, text) pairs, reusing cached vectors."""
        to_embed: list[tuple[int, str, str]] = []
        resolved: list[tuple[int, str, np.ndarray]] = []
        for eid, text in items:
            h = text_hash(text)
            cached = self._db.find_embedding_by_hash(self._embedder.name, h)
            if cached is not None:
                self.cache_reuses += 1
                resolved.append((eid, h, np.frombuffer(cached, dtype=np.float32)))
            else:
                to_embed.append((eid, text, h))
        if to_embed:
            self.embed_calls += 1
            vecs = self._embedder.encode([t for _, t, _ in to_embed])
            resolved.extend((eid, h, vecs[i]) for i, (eid, _, h) in enumerate(to_embed))
        with self._lock:
            for eid, h, vec in resolved:
                self._db.put_embedding(eid, self._embedder.name, h, vec.astype(np.float32).tobytes(), self._embedder.dim)
                self._ids.append(eid)
            if resolved:
                self._matrix = np.vstack([self._matrix] + [v[None, :] for _, _, v in resolved])

    def remove(self, ids: set[int]) -> None:
        with self._lock:
            keep = [i for i, eid in enumerate(self._ids) if eid not in ids]
            self._ids = [self._ids[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self._embedder.dim), dtype=np.float32)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        with self._lock:
            if not self._ids:
                return []
            sims = self._matrix @ query_vec
            k = min(k, len(self._ids))
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            return [(self._ids[i], float(sims[i])) for i in top]

    def similarity(self, query_vec: np.ndarray, ids: Sequence[int]) -> dict[int, float]:
        with self._lock:
            pos = {eid: i for i, eid in enumerate(self._ids)}
            rows = [pos[i] for i in ids if i in pos]
            if not rows:
                return {}
            sims = self._matrix[rows] @ query_vec
            return {self._ids[r]: float(s) for r, s in zip(rows, sims)}
