"""Composition root: the one place that knows concrete classes.

Everything else depends on protocols; this module picks implementations and
wires them. Later phases add capabilities by changing *this* file's wiring,
not the orchestrator or agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agents.agent import Agent
from .agents.pipeline import Pipeline
from .config import Config
from .core.events import EventBus
from .context.tracker import ContextTracker
from .core.interfaces import ModelProvider, TokenCounter, ToolBridge
from .core.orchestrator import Orchestrator
from .llm.prompts import DEFAULT_SPECS, AgentSpec
from .llm.router import Router
from .memory.active_context import ActiveContext
from .memory.associative_graph import AssociativeGraph
from .memory.database import Database
from .memory.embedding_service import Embedder, EmbeddingService, HashingEmbedder, MiniLMEmbedder
from .memory.manager import MemoryManager
from .memory.retrieval import Retriever
from .memory.semantic_cache import SemanticCache
from .skills.null_bridge import NullToolBridge


@dataclass
class App:
    cfg: Config
    model: ModelProvider
    events: EventBus
    orchestrator: Orchestrator
    memory: MemoryManager | None = None
    db: Database | None = None


def make_embedder(cfg: Config) -> Embedder:
    if cfg.embedder == "hashing":
        return HashingEmbedder()
    return MiniLMEmbedder(cfg.embedding_model)


def build_memory(cfg: Config, db: Database, embedder: Embedder) -> MemoryManager:
    emb = EmbeddingService(embedder, db)
    graph = AssociativeGraph(db, decay=cfg.graph_decay, max_hops=cfg.graph_max_hops)
    cache = SemanticCache()
    return MemoryManager(db, emb, graph, Retriever(db, emb, graph, cache), cache)


def build_app(
    cfg: Config,
    model: ModelProvider,
    *,
    bridge: ToolBridge | None = None,
    events: EventBus | None = None,
    specs: dict[str, AgentSpec] | None = None,
    db: Database | None = None,
    embedder: Embedder | None = None,
    checkpoints=None,
    on_reply=None,
    file_reader=None,
) -> App:
    events = events or EventBus()
    specs = specs or DEFAULT_SPECS
    tracker = ContextTracker(model, cfg.context_recovery_threshold) if isinstance(model, TokenCounter) else None
    agents = {name: Agent(spec, model, cfg, tracker) for name, spec in specs.items()}
    pipeline = Pipeline(agents, cfg, bridge or NullToolBridge(), events)
    router = Router(model, max_tokens=cfg.router_max_tokens, temperature=cfg.router_temperature)
    context = ActiveContext(cfg.active_message_limit)

    memory = None
    if cfg.memory_enabled and db is not None:
        memory = build_memory(cfg, db, embedder or make_embedder(cfg))
        memory.attach(context, cfg.active_message_limit)

    orch = Orchestrator(router, pipeline, context, events, memory=memory, recall_k=cfg.recall_k,
                        checkpoints=checkpoints, on_reply=on_reply, file_reader=file_reader)
    return App(cfg, model, events, orch, memory, db)


# --------------------------------------------------------------------- skills
def build_skill_engine(
    cfg: Config,
    model: ModelProvider,
    approver,
    *,
    events: EventBus | None = None,
    installer=None,
    runner=None,
    builtin_dir: Path | None = None,
    on_job=None,
):
    from .execution.artifacts import WorkspaceManager
    from .execution.process_runner import ProcessRunner
    from .skills.dependencies import DependencyManager, pip_installer
    from .skills.engine import SkillEngine
    from .skills.generator import SkillGenerator
    from .skills.recipes import RecipeBook
    from .skills.registry import SkillRegistry, SkillSource
    from .skills.resolver import SkillResolver

    builtin = builtin_dir or Path(__file__).parent / "skills" / "builtin"
    generated = Path(cfg.skill_directory)
    registry = SkillRegistry([SkillSource(builtin, "builtin"), SkillSource(generated, "generated")])
    recipes = RecipeBook() if cfg.recipes_enabled else None
    generator = (
        SkillGenerator(model, max_tokens=cfg.skill_gen_max_tokens) if cfg.skill_generation_enabled else None
    )
    deps = DependencyManager(approver, installer or pip_installer, auto_approve=cfg.auto_install_dependencies)
    return SkillEngine(
        registry,
        SkillResolver(registry, recipes),
        runner or ProcessRunner(timeout_s=cfg.skill_timeout_s),
        WorkspaceManager(cfg.working_directory),
        generated,
        approver,
        deps,
        generator,
        events=events,
        max_retries=cfg.max_skill_retries,
        on_job=on_job,
    )


# --------------------------------------------------------------- full system
@dataclass
class System:
    """Everything wired for real use: memory, skills, native tools, workflows, metrics."""

    app: App
    engine: object
    workflows: object
    metrics: object | None
    checkpoints: object
    vision: object


def make_vision(model):
    from .llm.vision import ModelVision, UnsupportedVision

    if hasattr(model, "vision_status") and hasattr(model, "describe_image"):
        return ModelVision(model)
    return UnsupportedVision("the model backend has no image input")


def build_system(
    cfg: Config,
    model: ModelProvider,
    approver,
    *,
    db: Database | None = None,
    embedder: Embedder | None = None,
    events: EventBus | None = None,
    gui_driver=None,
    project_root: Path | None = None,
) -> System:
    from .context.checkpoint import CheckpointStore
    from .context.project_builder import ProjectBuilder
    from .context.recovery import LongCodeGenerator
    from .core.metrics import InstrumentedModel, MetricsStore
    from .core.workflow import WorkflowRunner
    from .execution.artifacts import WorkspaceManager
    from .execution.desktop_loop import DesktopLoop
    from .execution.gui_adapter import PyAutoGuiDriver
    from .tools.codegen import BuildProjectTool
    from .tools.file_builder import AliasTool
    from .tools.desktop import DesktopTaskTool, LookAtScreenTool
    from .tools.external import ChatWithAppTool, FindChatAppTool
    from .tools.file_builder import CreateFileTool, SectionedFileBuilder
    from .tools.html_builder import HtmlPageBuilder
    from .tools.files import WriteFileTool
    from .tools.self_improve import ImproveSourceTool, SourceModifier

    events = events or EventBus()
    db = db or Database(cfg.database_path)
    metrics = MetricsStore(db) if cfg.metrics_enabled else None
    if metrics is not None:
        model = InstrumentedModel(model, metrics)  # wraps the one model; never a second instance

    checkpoints = CheckpointStore(cfg.checkpoint_directory)
    workspaces = WorkspaceManager(cfg.working_directory)

    def on_job(job):
        if metrics is not None:
            metrics.record("skill_success", 1.0 if job.status == "success" else 0.0, level=str(job.level))
            metrics.record("skill_attempts", job.attempt, level=str(job.level))

    engine = build_skill_engine(cfg, model, approver, events=events, on_job=on_job)

    tracker = ContextTracker(model, cfg.context_recovery_threshold)
    longgen = LongCodeGenerator(model, tracker, checkpoints, events=events)
    writer = WriteFileTool(cfg.output_directory, approver)
    engine.registry.register_native(writer)
    file_tracker = ContextTracker(model, cfg.context_recovery_threshold)
    builder = SectionedFileBuilder(model, file_tracker, events=events)
    pages = HtmlPageBuilder(model, file_tracker, events=events)
    from .core.file_refs import FileReferenceReader

    create = CreateFileTool(builder, writer, pages, file_reader=FileReferenceReader(base_dirs=(Path.cwd(),)))
    engine.registry.register_native(create)
    out_dir = writer.output_dir
    # One file tool: the old name is kept as a shortcut so every file gets the full checks.
    engine.registry.register_native(AliasTool("write_python_module", "Same as create_file, for a Python file", create))
    engine.registry.register_native(BuildProjectTool(ProjectBuilder(model, longgen, events=events), workspaces, out_dir))
    engine.registry.register_native(FindChatAppTool())
    engine.registry.register_native(ChatWithAppTool(model, approver, events))

    vision = make_vision(model)
    if cfg.gui_enabled:
        driver = gui_driver or PyAutoGuiDriver()
        engine.registry.register_native(LookAtScreenTool(driver, vision, workspaces))
        loop = DesktopLoop(driver, vision, approver, events=events, max_steps=cfg.desktop_max_steps)
        engine.registry.register_native(DesktopTaskTool(loop, workspaces))

    if cfg.self_modification_enabled:
        root = project_root or Path(__file__).resolve().parent.parent
        modifier = SourceModifier(root, Path(cfg.checkpoint_directory) / "source", approver,
                                  hot_dirs=(Path(cfg.skill_directory),))
        engine.registry.register_native(ImproveSourceTool(model, modifier))

    def on_reply(reply):
        if metrics is not None and reply.route is not None:
            metrics.record("turn_seconds", reply.elapsed_s, route=reply.route.label())
            metrics.record("router_clean_parse", 1.0 if reply.route.parsed_cleanly else 0.0)
            metrics.record("turn_model_calls", reply.model_calls, route=reply.route.label())

    app = build_app(cfg, model, bridge=engine, events=events, db=db, embedder=embedder,
                    checkpoints=checkpoints, on_reply=on_reply,
                    file_reader=FileReferenceReader(base_dirs=(Path.cwd(), out_dir)))
    workflows = WorkflowRunner(model, app.orchestrator.pipeline, checkpoints, approver, events=events)
    return System(app, engine, workflows, metrics, checkpoints, vision)
