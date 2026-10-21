"""SQLite persistence — the authoritative store for long-term memory.

SRP: schema, connections and small typed queries. Retrieval logic lives in
retrieval.py; graph logic in associative_graph.py. Both go through this.

Schema
  messages        every user/assistant message (in_window marks the live FIFO)
  evidence        retrievable units (sentences of archived messages, facts, source summaries)
  embeddings      one vector per evidence row, keyed by model + text hash
  nodes           graph terms (normalised labels)
  node_evidence   term → evidence references (the graph never copies text)
  node_edges      term ↔ term co-occurrence counts
  metrics         Phase 7 measurements
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, role TEXT NOT NULL, content TEXT NOT NULL,
    created_at REAL NOT NULL, in_window INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
    last_accessed REAL, access_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS embeddings (
    evidence_id INTEGER PRIMARY KEY REFERENCES evidence(id) ON DELETE CASCADE,
    model TEXT NOT NULL, dim INTEGER NOT NULL, text_hash TEXT NOT NULL, vector BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS emb_hash ON embeddings(model, text_hash);
CREATE TABLE IF NOT EXISTS nodes (id INTEGER PRIMARY KEY, label TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS node_evidence (
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    evidence_id INTEGER NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    PRIMARY KEY (node_id, evidence_id));
CREATE INDEX IF NOT EXISTS ne_evidence ON node_evidence(evidence_id);
CREATE TABLE IF NOT EXISTS node_edges (
    a INTEGER NOT NULL, b INTEGER NOT NULL, weight REAL NOT NULL DEFAULT 1,
    PRIMARY KEY (a, b));
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, name TEXT NOT NULL,
    value REAL NOT NULL, tags TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS metrics_name ON metrics(name);
"""


@dataclass(frozen=True)
class EvidenceRow:
    id: int
    kind: str
    text: str
    source_ref: str
    created_at: float
    access_count: int


class Database:
    """Thread-safe wrapper: one connection, serialised by a lock.

    The console runs tasks on a worker thread, so check_same_thread is off and
    every statement goes through `_lock`.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if str(path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            self._conn.commit()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- messages
    def add_message(self, role: str, content: str) -> int:
        with self.tx() as c:
            return c.execute(
                "INSERT INTO messages(role, content, created_at) VALUES (?, ?, ?)", (role, content, time.time())
            ).lastrowid

    def mark_archived(self, message_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE messages SET in_window = 0 WHERE id = ?", (message_id,))

    def window_messages(self, limit: int) -> list[sqlite3.Row]:
        rows = self.query(
            "SELECT id, role, content FROM messages WHERE in_window = 1 ORDER BY id DESC LIMIT ?", (limit,)
        )
        return list(reversed(rows))

    # ---------------------------------------------------------------- evidence
    def add_evidence(self, kind: str, text: str, source_ref: str = "") -> int:
        with self.tx() as c:
            return c.execute(
                "INSERT INTO evidence(kind, text, source_ref, created_at) VALUES (?, ?, ?, ?)",
                (kind, text, source_ref, time.time()),
            ).lastrowid

    def get_evidence(self, ids: Iterable[int]) -> dict[int, EvidenceRow]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.query(
            f"SELECT id, kind, text, source_ref, created_at, access_count FROM evidence WHERE id IN ({marks})", ids
        )
        return {r["id"]: EvidenceRow(**dict(r)) for r in rows}

    def touch_evidence(self, ids: Iterable[int]) -> None:
        now = time.time()
        with self.tx() as c:
            c.executemany(
                "UPDATE evidence SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                [(now, i) for i in ids],
            )

    def delete_evidence(self, ids: Iterable[int]) -> None:
        with self.tx() as c:
            c.executemany("DELETE FROM evidence WHERE id = ?", [(i,) for i in ids])

    def evidence_count(self) -> int:
        return self.query("SELECT COUNT(*) AS n FROM evidence")[0]["n"]

    # -------------------------------------------------------------- embeddings
    def put_embedding(self, evidence_id: int, model: str, text_hash: str, vector: bytes, dim: int) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO embeddings(evidence_id, model, dim, text_hash, vector) VALUES (?,?,?,?,?)",
                (evidence_id, model, dim, text_hash, vector),
            )

    def find_embedding_by_hash(self, model: str, text_hash: str) -> bytes | None:
        rows = self.query("SELECT vector FROM embeddings WHERE model = ? AND text_hash = ? LIMIT 1", (model, text_hash))
        return rows[0]["vector"] if rows else None

    def all_embeddings(self, model: str) -> list[sqlite3.Row]:
        return self.query("SELECT evidence_id, dim, vector FROM embeddings WHERE model = ? ORDER BY evidence_id", (model,))

    # ------------------------------------------------------------------ metrics
    def add_metric(self, name: str, value: float, tags: str = "") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO metrics(ts, name, value, tags) VALUES (?,?,?,?)", (time.time(), name, value, tags))
