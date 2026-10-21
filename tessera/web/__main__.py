"""python -m tessera.web  — Tessera in your browser.

Run from the folder that holds tessera.toml and models/ (e.g. D:\\tessera), or pass --home.
Close the console Tessera first: only one copy of the model fits in GPU memory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from ..__main__ import load_model, scripted_demo_model
    from ..app import build_system
    from ..config import load_config
    from .server import EventHub, WebApprover, serve

    ap = argparse.ArgumentParser(prog="python -m tessera.web")
    ap.add_argument("--home", default=None, help="folder with tessera.toml and models/ (default: current folder, "
                                                 "or the Tessera project folder if it has a tessera.toml)")
    ap.add_argument("--config", default="tessera.toml")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless you know why (Tessera can write files)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    ap.add_argument("--scripted", action="store_true", help="run without a model (canned replies)")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    home = Path(args.home) if args.home else _guess_home(args.config)
    os.chdir(home)   # relative paths in tessera.toml (models/…, data/…) resolve exactly like the console app
    cfg = load_config(args.config)
    if args.scripted:
        cfg = cfg.with_overrides(embedder="hashing", skill_generation_enabled=False)

    try:
        model = scripted_demo_model() if args.scripted else load_model(cfg)
    except Exception as exc:
        print(f"Could not load the model: {exc}", file=sys.stderr)
        return 1

    hub = EventHub()
    approver = WebApprover(hub)
    system = build_system(cfg, model, approver)
    name = "scripted demo" if args.scripted else Path(str(getattr(cfg, "model_path", "") or cfg.backend)).name
    try:
        server = serve(system.app.orchestrator, system.app.events, approver, hub,
                       host=args.host, port=args.port, model_name=name)
    except OSError as exc:
        print(f"Could not open port {args.port}: {exc}. Try --port 8001.", file=sys.stderr)
        return 1

    url = f"http://localhost:{args.port}"
    print(f"Tessera is ready at {url}   (folder: {home}; Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
        close = getattr(model, "close", None)
        if callable(close):
            close()
    return 0


def _guess_home(config: str) -> Path:
    here = Path.cwd()
    if (here / config).exists():
        return here
    project = Path(__file__).resolve().parents[2]   # …/tessera (project) / tessera (package) / web
    return project if (project / config).exists() or (project / "models").is_dir() else here


if __name__ == "__main__":
    raise SystemExit(main())
