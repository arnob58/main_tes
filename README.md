# Tessera v0.1

A local-first agentic assistant for a 6 GB GPU. **Gemma reasons, Python executes, SQLite + MiniLM remember, the skill engine extends, the orchestrator coordinates.** All seven phases of the spec are implemented and tested against fakes for the model, screen and network; the parts that need your hardware are listed under *Not yet verified*.

See `ARCHITECTURE.md` for how the pieces fit.

## Install

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
# llama-cpp-python with CUDA (needs the CUDA toolkit). Version ≥ 0.3.36 has Gemma4ChatHandler (vision).
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir "llama-cpp-python>=0.3.36"
#   PowerShell: $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --no-cache-dir "llama-cpp-python>=0.3.36"
pip install -e ".[memory,desktop,dev]"     # sentence-transformers, pyautogui+pillow, pytest
```

**Model files** (put them in `models/`):
- an instruction-tuned **Gemma 4 E2B** GGUF — Q4_K_M is a sensible start for 6 GB
- its **mmproj** GGUF (vision projector) — needed only for screen understanding

MiniLM (`sentence-transformers/paraphrase-MiniLM-L3-v2`) downloads automatically on first use (~70 MB, CPU).

## Configure

```bash
cp tessera.example.toml tessera.toml
```
Set `model_path` and `mmproj_path`. Every field in `tessera/config.py` can be set there or as `TESSERA_<FIELD>` env vars.

**Two backends, same behaviour:**
- `backend = "llama_cpp"` (default) — model loaded in-process; vision via `Gemma4ChatHandler` on the same instance.
- `backend = "server"` — talk to llama.cpp's server, which gets new model support first:
  `llama-server -m models/gemma.gguf --mmproj models/mmproj.gguf -c 4096 -ngl 99 --port 8080`

## Run

```bash
python -m tessera                 # real model
python -m tessera --scripted      # no model: canned replies, to try the UI and plumbing
python scripts/smoke_test.py --model models/<file>.gguf    # first real-hardware check
python scripts/benchmark.py                                # Phase 7 measurements
python -m pytest -q                                        # 130 tests, no GPU needed
```

Console commands: `/remember`, `/recall`, `/forget`, `/skills`, `/workflow <objective>`, `/workflows`, `/resume <id>`, `/approve <id> <step>`, `/stats`, `/history`, `/debug`, `/quit`. Ctrl+C cancels a running task at the next safe point. Consequential actions (package installs, risky generated code, typing into apps, submit/send/apply clicks, running external apps, source patches) ask first; answer `a` to always allow that kind for the session.

## What each phase delivers

| Phase | Delivered | Key evidence in tests |
|---|---|---|
| 1 Core | single-model engine with exact token counts, GBNF-constrained router, data-driven agents and pipelines, console | one model instance shared by all agents; bounded loops; cancellation |
| 2 Memory | 5-message window, SQLite archive, MiniLM index, term graph, spreading activation, safe semantic cache | cookie → bakery → childhood → grandmother found with ~0 similarity |
| 3 Skills | AST-only discovery, L1 skill / L2 recipe / L3 generation, sandboxed child-process runtime, deps, repair, promotion | zero-skill bootstrap end-to-end; missing dependency installed and retried |
| 4 Context | real-tokenizer tracking, 90% trigger, AST split, checkpoints, fresh-context continuation, project builder | module 2.4× the window rebuilt byte-identical over 4 recoveries; crash resume |
| 5 Desktop | PyAutoGUI driver, normalised coordinates, multimodal observe-act loop, llama-server backend | job-application flow with approval gates; stale screenshots never clicked |
| 6 Autonomy | resumable workflows, task checkpoints, chatbot discovery + adapters, AI-to-AI dialogue, tested self-patching | "GPT Luna" found via `ast`, multi-turn chat with session state; failing patch rolled back |
| 7 Measure | per-stage latency/tokens/s, skill success, router accuracy, RSS/VRAM, benchmark script | metrics recorded through a model decorator that never loads a second model |

## Test report

**Executed here: 130 passed** (Python 3.13, Linux, no GPU, no display, scripted model). Real child processes, real SQLite, real HTTP servers on localhost, and the real installed `llama-cpp-python` 0.3.36 for prompt rendering, GBNF compilation and the vision handler class.

| File | Tests | Covers |
|---|---|---|
| test_config | 4 | defaults, TOML/env layering, unknown-key rejection |
| test_router | 10 | strict/lenient/fallback parsing, grammar + budget |
| test_engine | 7 | one-model rule, counted = evaluated tokens, clamping, KV reset |
| test_active_context | 2 | FIFO eviction |
| test_pipeline | 9 | stage shapes, handoffs, revision + pass bounds, skill round-trips |
| test_orchestrator | 8 | end-to-end turn, no internal state in memory, cancellation, OCP |
| test_memory | 11 | associative chain, provenance, persistence, cache invalidation, forget |
| test_skills | 25 | discovery, observation capture, L1/L2/L3, deps, repair, approvals, relative-path regression |
| test_context_recovery | 13 | tracker math, AST split invariants, recovery, resume, prompt fitting, project build |
| test_desktop_vision | 15 | loop + gates, coordinates, server client, real llama-cpp checks |
| test_autonomy | 15 | workflows, external adapters, AI conversation, self-patching |
| test_architecture | 11 | import-graph layering, wiring, metrics |

Bugs the tests caught during the build: plural normalisation broke graph links ("cookies" → "cooky"); continuation prompts grew with every function until they overflowed; relative workspaces broke the child process (found by running the console end to end, then pinned with a regression test).

## Not yet verified (needs your machine)

- **Any real Gemma output.** Prompt quality, router accuracy, skill-generation success and the action-JSON reliability of a 2B-class model are unmeasured. Run `scripts/smoke_test.py` then `scripts/benchmark.py`.
- **Gemma 4 E2B vision through llama-cpp-python** on your GPU (handler exists; not exercised with weights).
- **Real MiniLM** — tests use a lexical hashing embedder because model downloads were blocked here.
- **PyAutoGUI on a real display**, Chrome launching, HiDPI scaling.
- **CUDA build, VRAM headroom** with model + mmproj + 4k context on 6 GB.

## Known limitations

- Generated skills run as you, with your environment variables. The static risk scan catches common destructive calls; it is not a sandbox.
- Promotion after one successful run proves only that run; there is no regression suite per skill yet.
- GUI clicking relies on the model's coordinates; small vision models read screens better than they point.
- Core self-patches are applied on disk but only take effect after restart (reported as such). Skill patches are live immediately.
- Workflow planning quality depends on the model; plans are capped at 10 steps.
- The procedural/heuristic memory subsystem is intentionally not implemented (interface space reserved).
