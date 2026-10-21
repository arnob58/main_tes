"""Central configuration.

Defaults live here; overrides come from a TOML file and/or TESSERA_*
environment variables. No other module hard-codes these numbers.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

DEFAULT_PIPELINES: dict[str, list[str]] = {
    "instant": ["interpret", "respond"],
    "medium": ["interpret", "develop", "respond"],
    "complex": ["interpret", "develop", "evaluate", "respond"],
}


@dataclass(frozen=True)
class Config:
    # --- model -----------------------------------------------------------
    backend: str = "llama_cpp"     # "llama_cpp" (in-process) | "server" (llama-server over HTTP)
    server_url: str = "http://127.0.0.1:8080"
    model_path: str = "models/gemma-4-e2b-it-Q4_K_M.gguf"
    mmproj_path: str = ""          # vision projector, used from Phase 5
    gpu_layers: int = -1           # -1 = offload everything that fits
    context_window: int = 4096
    n_threads: int = 0             # 0 = let llama.cpp decide
    seed: int = 42
    reset_kv_between_calls: bool = False  # prefix reuse is safe; see engine docs

    # --- generation budgets ----------------------------------------------
    router_max_tokens: int = 8
    agent_max_tokens: int = 384
    respond_max_tokens: int = 512
    default_temperature: float = 0.3
    router_temperature: float = 0.0

    # --- reasoning -------------------------------------------------------
    pipelines: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_PIPELINES))
    max_agent_iterations: int = 8   # total agent passes allowed per request
    max_revisions: int = 2          # Evaluate → Develop loops
    max_skill_calls: int = 4        # Develop → skill → Develop loops

    # --- context / memory ------------------------------------------------
    context_recovery_threshold: float = 0.90
    active_message_limit: int = 5
    embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    database_path: str = "storage/database.sqlite"
    memory_enabled: bool = True
    embedder: str = "minilm"        # "minilm" | "hashing" (lexical stand-in, no download)
    recall_k: int = 5
    graph_max_hops: int = 3
    graph_decay: float = 0.6

    # --- skills ----------------------------------------------------------
    skill_directory: str = "tessera/skills/generated"   # promoted, reusable skills
    working_directory: str = "storage/working"         # per-task workspaces
    max_skill_retries: int = 3
    skill_timeout_s: float = 60.0
    skills_enabled: bool = True
    recipes_enabled: bool = True
    skill_generation_enabled: bool = True
    skill_gen_max_tokens: int = 900
    auto_install_dependencies: bool = False            # otherwise ask each time

    # --- autonomy (phases 5-6) --------------------------------------------
    checkpoint_directory: str = "storage/checkpoints"
    gui_enabled: bool = True
    desktop_max_steps: int = 15
    self_modification_enabled: bool = True
    metrics_enabled: bool = True

    def with_overrides(self, **overrides: Any) -> "Config":
        return replace(self, **overrides)


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Defaults < TOML file < environment variables (TESSERA_<FIELD>)."""
    cfg = Config()
    known = {f.name: f for f in fields(Config)}

    if path is not None and Path(path).exists():
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = replace(cfg, **data)

    env = os.environ if env is None else env
    updates: dict[str, Any] = {}
    for name in known:
        key = f"TESSERA_{name.upper()}"
        if key in env and not isinstance(getattr(cfg, name), (dict, list)):
            updates[name] = _coerce(env[key], getattr(cfg, name))
    return replace(cfg, **updates) if updates else cfg
