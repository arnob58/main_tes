# Tessera architecture

## Request flow

```
console ──handle(text)──▶ Orchestrator
                            │ 1. Router (1 short, grammar-constrained call) → Complexity|skills
                            │ 2. MemoryManager.recall → evidence block (labelled as leads)
                            │ 3. Pipeline: stages from config, e.g. interpret → develop → evaluate → respond
                            │       develop: "ACTION: name {json}" ──▶ ToolBridge (SkillEngine)
                            │                                            ├─ native tool (desktop, codegen, chat, self-patch)
                            │                                            └─ L1 skill / L2 recipe / L3 generate
                            │                                               → static risk scan → Approver
                            │                                               → child process → Observation
                            │                                               → recovery (install / fix args / model repair)
                            │                                               → promote on success
                            │       observation → develop again (bounded)
                            │       evaluate VERDICT: REVISE → develop again (bounded)
                            │ 4. user + final reply → ActiveContext (5) → evicted → archive/embed/graph
                            ▼
                          Reply (+ checkpoint on failure, metrics hook)
```

## Layers and their single responsibilities

| Package | Owns | Must not import |
|---|---|---|
| `core` | data types, protocols, events, task state, orchestrator, workflows, metrics | UI, tools, concrete engines |
| `llm` | inference backends, router, prompt specs, vision adapters, scripted model | skills, execution, memory, UI |
| `agents` | the generic Agent, the stage pipeline, ACTION/VERDICT parsers | skills, execution, engines, UI |
| `memory` | SQLite schema, embeddings, term graph, retrieval, cache, window | llm, agents, skills, UI |
| `context` | token tracking, AST analysis, checkpoints, continuation, project builder | skills, tools, UI |
| `skills` | contract, registry, resolver, recipes, generator, recovery, deps, policy, engine | agents, engines, tools, UI |
| `execution` | child-process runner + harness, workspaces, GUI driver, desktop loop | agents, skills, UI |
| `tools` | native tools that adapt services to the skill contract | — (leaf) |
| `ui` | console | — (leaf) |
| `app.py` | **composition root** — the only place concrete classes are chosen | — |

These rules are enforced by `tests/test_architecture.py`, which parses every import.

## Principles, concretely

- **SRP** — e.g. `ProcessRunner` only spawns and observes; `SkillRecovery` only classifies failures; `SkillEngine` only sequences them.
- **OCP** — new skills: drop a file in a scanned folder. New recipes: drop a file in `skills/recipes/`. New native tools: `register_native()` in `app.py`. New agent stages and pipeline shapes: `AgentSpec` + config. None of these touch the orchestrator, router, pipeline or engine (tested).
- **DIP** — orchestrator, router, agents and workflows depend on `ModelProvider`, `TokenCounter`, `ToolBridge`, `MemoryRetriever`, `Approver`, `EventSink`. llama.cpp in-process, llama-server, the scripted test model and the metrics decorator are interchangeable.
- **ISP** — a component that counts tokens depends on `TokenCounter` only; skills implement `run(**kwargs)` and nothing else.

## Key invariants

1. **One model.** `LlamaCppEngine` refuses a second live instance; `InstrumentedModel` wraps, never loads.
2. **Agent isolation.** Each call is a fresh message list; llama.cpp only reuses a token-identical KV prefix (verified in the library source).
3. **Counted = evaluated.** Prompts are rendered with the GGUF's template and tokenized once; those token ids are what run.
4. **Nothing internal is remembered.** Only user messages, final replies and explicit facts become memory.
5. **Discovery never executes.** Skill metadata comes from `ast.literal_eval`.
6. **Success needs evidence.** Exit 0 *and* a harness result; consequential workflow steps need a successful tool observation.
7. **Nothing is dropped.** `complete_prefix + incomplete_suffix == source` for every recovery split.
8. **Recovery is real.** Continuations are built only from disk state, and the KV cache is reset when the backend supports it (`fresh_context` is recorded per round).

## Storage

```
storage/database.sqlite      messages, evidence, embeddings, nodes, node_evidence, node_edges, metrics
storage/working/<task>/      per-task workspaces (generated code, artifacts, screenshots)
storage/checkpoints/         task_*, workflow_*, long-generation checkpoints; source/ = patch backups
tessera/skills/generated/    promoted, reusable skills
```

## Reserved extension points

- `MemoryRetriever` can gain a procedural/heuristic memory implementation without orchestrator changes.
- `GuiDriver` can gain an accessibility-tree/OCR driver that returns element boxes (more precise than model coordinates).
- `ToolBridge` can be composed (e.g. a remote tool host) since the pipeline only sees the protocol.
