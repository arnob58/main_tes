"""Entry point: `python -m tessera [--config tessera.toml] [--scripted]`."""

from __future__ import annotations

import argparse
import logging
import sys

from .app import build_system
from .config import load_config
from .llm.scripted import ScriptedModel
from .ui.console import ConsoleApprover, ConsoleUI


def scripted_demo_model() -> ScriptedModel:
    """Offline stand-in so the UI and control flow can be tried without a GPU.
    Its replies are canned and say so — they are not model output."""

    def request_of(messages) -> str:
        body = messages[-1].content
        return body.split("REQUEST:\n", 1)[-1].split("\n\n", 1)[0]

    def route(messages) -> str:
        text = messages[-1].content.lower()
        if any(w in text for w in ("file", "find", "open", "run", "code", "calculate")):
            return "Complex|yes"
        if len(text.split()) > 12:
            return "Medium|no"
        return "Instant|no"

    def develop(messages) -> str:
        body = messages[-1].content
        if "OBSERVATIONS:" in body:
            return "Draft based on the observation above."
        if "calculate" in request_of(messages).lower():
            expr = request_of(messages).lower().split("calculate", 1)[1].strip(" ?.")
            return f'ACTION: calculator {{"expression": "{expr}"}}'
        return f"Draft answer for: {request_of(messages)}"

    return (
        ScriptedModel()
        .when_agent("router", route)
        .when_agent("interpret", lambda m: f"INTENT: respond to '{request_of(m)}'\nTOOLS: none")
        .when_agent("develop", develop)
        .when_agent("evaluate", "No problems found.\nVERDICT: PASS")
        .when_agent("respond", lambda m: "[scripted model] " + (
            m[-1].content.split("OBSERVATIONS:\n", 1)[1].split("\n\n", 1)[0] if "OBSERVATIONS:" in m[-1].content
            else f"I received: {request_of(m)}"))
    )


def load_model(cfg):
    if cfg.backend == "server":
        from .llm.server_backend import LlamaServerEngine

        print(f"Connecting to llama-server at {cfg.server_url}…", flush=True)
        return LlamaServerEngine(cfg.server_url)
    from .llm.engine import LlamaCppEngine

    print(f"Loading {cfg.model_path} (gpu_layers={cfg.gpu_layers}, n_ctx={cfg.context_window})…", flush=True)
    return LlamaCppEngine(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tessera")
    ap.add_argument("--config", default="tessera.toml", help="TOML overrides (optional)")
    ap.add_argument("--scripted", action="store_true", help="run without a model (canned replies)")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.scripted:
        cfg = cfg.with_overrides(embedder="hashing", skill_generation_enabled=False)

    try:
        model = scripted_demo_model() if args.scripted else load_model(cfg)
    except Exception as exc:
        print(f"Could not load the model: {exc}", file=sys.stderr)
        return 1

    approver = ConsoleApprover(input, sys.stdout)
    system = build_system(cfg, model, approver)
    try:
        ConsoleUI(system.app.orchestrator, system.app.events, system=system).loop()
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
