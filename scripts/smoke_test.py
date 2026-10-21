"""Real-hardware check for Phase 1. Run on the target machine:

    python scripts/smoke_test.py --model models/gemma-4-e2b-it-Q4_K_M.gguf

It prints facts it observed; it does not judge answer quality. Paste the
output back so the next phase can be tuned against real numbers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tessera.app import build_app  # noqa: E402
from tessera.config import load_config  # noqa: E402
from tessera.core.types import Message  # noqa: E402
from tessera.llm.engine import LlamaCppEngine  # noqa: E402
from tessera.llm.router import Router  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()
    cfg = load_config(None).with_overrides(model_path=args.model, context_window=args.ctx)

    t0 = time.perf_counter()
    engine = LlamaCppEngine(cfg)
    print(f"load_s={time.perf_counter() - t0:.1f}")

    msgs = [Message.system("Answer in one short sentence."), Message.user("What is 2 + 2?")]
    counted = engine.count_prompt_tokens(msgs)
    t0 = time.perf_counter()
    r = engine.generate(msgs, max_output_tokens=32)
    dt = time.perf_counter() - t0
    print(f"count_prompt_tokens={counted} runtime_prompt_tokens={r.input_tokens} match={counted == r.input_tokens}")
    print(f"reply={r.text!r} out_tokens={r.output_tokens} finish={r.finish_reason} tok_per_s={r.output_tokens / dt:.1f}")

    router = Router(engine)
    for q in ["hi", "explain photosynthesis step by step", "find my resume and summarize it"]:
        t0 = time.perf_counter()
        d = router.route(q)
        print(f"route[{q!r}] -> {d.label()} clean={d.parsed_cleanly} raw={d.raw!r} ms={1000 * (time.perf_counter() - t0):.0f}")

    app = build_app(cfg, engine)
    for q in ["Hello.", "Explain why the sky is blue in three sentences."]:
        reply = app.orchestrator.handle(q)
        print(f"\n[{reply.route.label() if reply.route else '-'}] calls={reply.model_calls} "
              f"in/out={reply.input_tokens}/{reply.output_tokens} {reply.elapsed_s:.1f}s\n{reply.text}")

    try:
        import llama_cpp
        print(f"\nllama_cpp version={getattr(llama_cpp, '__version__', '?')}")
        print("gpu_offload_supported=", bool(llama_cpp.llama_supports_gpu_offload()))
    except Exception as exc:  # informational only
        print(f"version check failed: {exc}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
