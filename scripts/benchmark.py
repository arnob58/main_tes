"""Phase 7 measurements on the target machine. Measures; never guesses.

    python scripts/benchmark.py --config tessera.toml          # real model
    python scripts/benchmark.py --scripted                     # plumbing check only

Reports:
  * router accuracy on a small labelled set (+ clean-parse rate)
  * per-stage latency and output tokens/s for a few end-to-end turns
  * skill path: Level-1 builtin call and Level-2 recipe call wall time
  * memory retrieval: does the associative chain surface the target?
  * process RSS and GPU memory (nvidia-smi) before and after
Writes everything to the metrics table and prints a report. In --scripted mode
every number is labelled SCRIPTED — those numbers say nothing about Gemma.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessera.app import build_system  # noqa: E402
from tessera.config import load_config  # noqa: E402
from tessera.core.approval import StaticApprover  # noqa: E402
from tessera.core.metrics import resource_snapshot  # noqa: E402
from tessera.core.types import SkillRequest  # noqa: E402
from tessera.llm.router import Router  # noqa: E402

ROUTER_SET = [
    ("hello!", "Instant|no"), ("thanks, that helps", "Instant|no"), ("what's 17% of 2,340?", "Instant|yes"),
    ("what day is it today?", "Instant|yes"), ("explain how a heat pump works", "Medium|no"),
    ("compare TCP and UDP for game networking", "Medium|no"), ("summarise the pdf on my desktop", "Complex|yes"),
    ("write a python script that renames my photos by date and run it", "Complex|yes"),
    ("design a database schema for a library app", "Complex|no"), ("open chrome and search for weather in toronto", "Complex|yes"),
]
TURNS = ["Hello.", "Explain in three sentences why the sky is blue.", "Calculate 2**20 / 7 and explain the result."]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tessera.toml")
    ap.add_argument("--scripted", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="tessera-bench-"))
    cfg = load_config(args.config).with_overrides(
        database_path=str(tmp / "bench.sqlite"), working_directory=str(tmp / "work"),
        checkpoint_directory=str(tmp / "ck"), skill_directory=str(tmp / "skills"), gui_enabled=False,
    )
    label = "SCRIPTED (not Gemma)" if args.scripted else "REAL MODEL"
    before = resource_snapshot()
    t0 = time.perf_counter()
    if args.scripted:
        from tessera.__main__ import scripted_demo_model

        cfg = cfg.with_overrides(embedder="hashing", skill_generation_enabled=False)
        model = scripted_demo_model()
    else:
        from tessera.__main__ import load_model

        model = load_model(cfg)
    load_s = time.perf_counter() - t0
    system = build_system(cfg, model, StaticApprover(True))
    m = system.metrics
    after_load = resource_snapshot()

    # router accuracy
    router = Router(system.app.model, max_tokens=cfg.router_max_tokens, temperature=cfg.router_temperature)
    correct = clean = 0
    rows = []
    for text, want in ROUTER_SET:
        d = router.route(text)
        ok = d.label() == want
        correct += ok
        clean += d.parsed_cleanly
        rows.append((text, want, d.label(), ok))
        m.record("bench_router_correct", 1.0 if ok else 0.0)

    # end-to-end turns
    turn_rows = []
    for t in TURNS:
        r = system.app.orchestrator.handle(t)
        turn_rows.append((t, r.route.label() if r.route else "-", r.model_calls, round(r.elapsed_s, 2), r.text[:100]))

    # skill paths
    eng = system.engine
    t = time.perf_counter()
    l1 = eng.execute(SkillRequest("calculator", {"expression": "3**7"}, task_id="bench"))
    l1_ms = (time.perf_counter() - t) * 1000
    (tmp / "docs").mkdir()
    (tmp / "docs" / "cv_resume.txt").write_text("resume")
    t = time.perf_counter()
    l2 = eng.execute(SkillRequest("search_files", {"root": str(tmp / "docs"), "name_contains": "resume"}, task_id="bench"))
    l2_ms = (time.perf_counter() - t) * 1000
    m.record("bench_skill_l1_ms", l1_ms)
    m.record("bench_skill_l2_ms", l2_ms)

    # memory chain
    mem = system.app.memory
    for s in ["The cookie smell drifted out of the kitchen.", "That cookie smell was just like the old bakery downtown.",
              "The bakery was where I spent much of my childhood.", "During childhood I visited my grandmother every Sunday.",
              "Quarterly taxes are due in April."]:
        mem.remember(s)
    hits = mem.recall("cookie smell", k=5)
    chain_ok = any("grandmother" in h.text for h in hits)
    m.record("bench_memory_chain_hit", 1.0 if chain_ok else 0.0)
    end = resource_snapshot()

    print(f"\n=== Tessera benchmark — {label} ===")
    print(f"model load: {load_s:.1f}s   backend={cfg.backend}   n_ctx={getattr(system.app.model, 'context_window', '?')}")
    print(f"\nRouter: {correct}/{len(ROUTER_SET)} correct, {clean}/{len(ROUTER_SET)} parsed cleanly")
    for text, want, got, ok in rows:
        print(f"  {'✓' if ok else '✗'} {got:<12} (want {want:<12}) {text}")
    print("\nTurns:")
    for t, route, calls, secs, text in turn_rows:
        print(f"  [{route}] {calls} calls {secs}s  {t!r} → {text!r}")
    print(f"\nSkills: L1 calculator {l1.status} {l1_ms:.0f} ms · L2 recipe {l2.status} {l2_ms:.0f} ms")
    print(f"Memory: associative chain {'reached' if chain_ok else 'did NOT reach'} the target "
          f"(embedder={cfg.embedder}); top hits: {[h.via for h in hits[:3]]}")
    fmt = lambda r: f"RSS {r.rss_mb:.0f} MB, GPU {r.gpu_used_mb:.0f}/{r.gpu_total_mb:.0f} MB" if r.gpu_used_mb is not None \
        else f"RSS {r.rss_mb:.0f} MB, GPU n/a ({r.note})" if r.rss_mb else f"GPU n/a ({r.note})"
    print(f"Resources: start {fmt(before)} · after load {fmt(after_load)} · end {fmt(end)}")
    print("\nPer-stage metrics:\n" + m.report())
    out = tmp / "benchmark.json"
    out.write_text(json.dumps({"label": label, "router": rows, "turns": turn_rows, "metrics": m.summary()}, indent=2, default=str))
    print(f"\nRaw results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
