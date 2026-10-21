"""Measurement, so optimisation follows data rather than guesses.

SRP split:
  MetricsStore      append/aggregate named measurements in SQLite
  InstrumentedModel ModelProvider decorator: per-stage latency, tokens, tokens/s
  resource_snapshot process RSS and (if nvidia-smi exists) GPU memory

InstrumentedModel wraps the single model instance — it never loads another.
Attribute access falls through to the wrapped engine, so reset_context(),
describe_image() etc. keep working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence

from ..memory.database import Database
from .types import GenerationResult, Message


class MetricsStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def record(self, name: str, value: float, **tags: str) -> None:
        self._db.add_metric(name, float(value), ",".join(f"{k}={v}" for k, v in sorted(tags.items())))

    def summary(self) -> dict[str, dict[str, float]]:
        rows = self._db.query(
            "SELECT name, COUNT(*) AS n, AVG(value) AS mean, MIN(value) AS lo, MAX(value) AS hi, SUM(value) AS total "
            "FROM metrics GROUP BY name ORDER BY name"
        )
        return {r["name"]: {"n": r["n"], "mean": r["mean"], "min": r["lo"], "max": r["hi"], "total": r["total"]} for r in rows}

    def by_tag(self, name: str, tag: str) -> dict[str, dict[str, float]]:
        rows = self._db.query("SELECT tags, value FROM metrics WHERE name = ?", (name,))
        groups: dict[str, list[float]] = {}
        for r in rows:
            key = dict(p.split("=", 1) for p in r["tags"].split(",") if "=" in p).get(tag, "?")
            groups.setdefault(key, []).append(r["value"])
        return {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in groups.items()}

    def report(self) -> str:
        s = self.summary()
        if not s:
            return "No measurements yet."
        lines = []
        for name, v in s.items():
            lines.append(f"{name:<32} n={v['n']:<5} mean={v['mean']:.3f}  min={v['min']:.3f}  max={v['max']:.3f}")
        for stage, v in sorted(self.by_tag("llm_latency_ms", "stage").items()):
            lines.append(f"  latency[{stage}]".ljust(34) + f"n={v['n']:<5} mean={v['mean']:.0f} ms")
        return "\n".join(lines)


def stage_of(messages: Sequence[Message]) -> str:
    if messages and messages[0].content.startswith("Stage: "):
        return messages[0].content.split("\n", 1)[0][7:].rstrip(".")
    return "other"


class InstrumentedModel:
    def __init__(self, inner, metrics: MetricsStore) -> None:
        self._inner = inner
        self._metrics = metrics

    def __getattr__(self, name):  # reset_context, describe_image, vision_status, close, …
        return getattr(self._inner, name)

    @property
    def context_window(self) -> int:
        return self._inner.context_window

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        return self._inner.count_prompt_tokens(messages)

    def generate(self, messages: Sequence[Message], **kw) -> GenerationResult:
        stage = stage_of(messages)
        t0 = time.perf_counter()
        try:
            r = self._inner.generate(messages, **kw)
        except Exception:
            self._metrics.record("llm_errors", 1, stage=stage)
            raise
        ms = (time.perf_counter() - t0) * 1000
        self._metrics.record("llm_latency_ms", ms, stage=stage)
        self._metrics.record("llm_input_tokens", r.input_tokens, stage=stage)
        self._metrics.record("llm_output_tokens", r.output_tokens, stage=stage)
        if ms > 0 and r.output_tokens:
            self._metrics.record("llm_output_tok_per_s", r.output_tokens / (ms / 1000), stage=stage)
        if r.finish_reason == "length":
            self._metrics.record("llm_hit_length_limit", 1, stage=stage)
        return r


@dataclass(frozen=True)
class ResourceSnapshot:
    rss_mb: float | None
    gpu_used_mb: float | None
    gpu_total_mb: float | None
    note: str = ""


def resource_snapshot() -> ResourceSnapshot:
    rss = None
    try:
        if sys.platform.startswith("linux"):
            with open(f"/proc/{os.getpid()}/status") as fh:
                rss = next(int(l.split()[1]) / 1024 for l in fh if l.startswith("VmRSS:"))
        else:
            import psutil  # optional

            rss = psutil.Process().memory_info().rss / 2**20
    except Exception:
        rss = None
    used = total = None
    note = ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            used, total = (float(x) for x in out.split(","))
        except Exception as exc:
            note = f"nvidia-smi failed: {exc}"
    else:
        note = "nvidia-smi not found"
    return ResourceSnapshot(rss, used, total, note)
