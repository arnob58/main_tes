"""llama.cpp inference backend (via llama-cpp-python).

SRP: load one GGUF model and run bounded, stateless chat completions with
exact token accounting. Nothing here knows about agents, routing or skills.

Design notes
------------
* Exactly one instance: constructing a second live engine raises
  `ModelAlreadyLoadedError`. Every logical agent shares this object.
* Exact token counts: the conversation is rendered with the GGUF's own Jinja
  chat template, tokenized with the model's tokenizer, and that *same token
  list* is what gets evaluated. The count we report is therefore the count
  the model saw — no character-based estimates.
* Agent isolation: llama-cpp-python keeps the KV cache from the previous call
  and reuses it only for the longest token-identical prefix of the new
  prompt. That is mathematically equivalent to a fresh evaluation, so one
  agent cannot leak state into another. Set `reset_kv_between_calls=True` to
  force a full reset anyway (slower, useful for debugging).
* Grammar: GBNF strings are compiled once and cached.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Sequence

from ..config import Config
from ..core.types import GenerationResult, Message

log = logging.getLogger("tessera.llm")


class ModelAlreadyLoadedError(RuntimeError):
    pass


class ContextOverflowError(RuntimeError):
    def __init__(self, prompt_tokens: int, context_window: int):
        super().__init__(
            f"Prompt needs {prompt_tokens} tokens but the context window is {context_window}."
        )
        self.prompt_tokens = prompt_tokens
        self.context_window = context_window


def vision_handler_for(cfg: Config) -> tuple[Any, str]:
    """(chat_handler or None, reason). Gemma 4 vision needs llama-cpp-python's
    Gemma4ChatHandler plus the model's mmproj projector file."""
    import os

    if not cfg.mmproj_path:
        return None, "no mmproj_path configured"
    if not os.path.exists(cfg.mmproj_path):
        return None, f"mmproj file not found: {cfg.mmproj_path}"
    try:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
    except ImportError:
        return None, "installed llama-cpp-python has no Gemma4ChatHandler (upgrade it, or use backend = 'server')"
    return Gemma4ChatHandler(clip_model_path=cfg.mmproj_path, verbose=False, use_gpu=cfg.gpu_layers != 0), "ok"


def _default_llama_factory(cfg: Config) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "llama-cpp-python is not installed. See README → Installation "
            "(build it with CUDA for GPU offload)."
        ) from exc
    handler, _ = vision_handler_for(cfg)
    return Llama(
        chat_handler=handler,
        model_path=cfg.model_path,
        n_gpu_layers=cfg.gpu_layers,
        n_ctx=cfg.context_window,
        n_threads=cfg.n_threads or None,
        seed=cfg.seed,
        verbose=False,
    )


def _default_grammar_compiler(gbnf: str) -> Any:
    from llama_cpp import LlamaGrammar  # pragma: no cover - needs llama_cpp

    return LlamaGrammar.from_string(gbnf, verbose=False)


def _default_formatter_factory(template: str, bos: str, eos: str) -> Callable[..., Any]:
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter  # pragma: no cover

    return Jinja2ChatFormatter(template=template, bos_token=bos, eos_token=eos)


class LlamaCppEngine:
    """ModelProvider + TokenCounter backed by a single llama.cpp model."""

    _live_instance: "LlamaCppEngine | None" = None
    _class_lock = threading.Lock()

    MIN_OUTPUT_TOKENS = 1

    def __init__(
        self,
        cfg: Config,
        *,
        llama_factory: Callable[[Config], Any] = _default_llama_factory,
        grammar_compiler: Callable[[str], Any] = _default_grammar_compiler,
        formatter_factory: Callable[[str, str, str], Callable[..., Any]] = _default_formatter_factory,
    ) -> None:
        with LlamaCppEngine._class_lock:
            if LlamaCppEngine._live_instance is not None:
                raise ModelAlreadyLoadedError(
                    "A model is already loaded; share that engine instead of loading another."
                )
            LlamaCppEngine._live_instance = self

        self._cfg = cfg
        self._lock = threading.Lock()  # llama.cpp contexts are not re-entrant
        self._grammar_compiler = grammar_compiler
        self._grammar_cache: dict[str, Any] = {}
        try:
            self._llm = llama_factory(cfg)
            self._bos_id, self._eos_id = self._llm.token_bos(), self._llm.token_eos()
            bos_text = self._token_text(self._bos_id)
            eos_text = self._token_text(self._eos_id)
            template = (getattr(self._llm, "metadata", {}) or {}).get("tokenizer.chat_template")
            if not template:
                raise RuntimeError(
                    "This GGUF has no embedded chat template; exact prompt rendering is impossible."
                )
            self._format = formatter_factory(template, bos_text, eos_text)
            self._bos_text = bos_text
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------ utils
    def _token_text(self, token_id: int) -> str:
        if token_id is None or token_id < 0:
            return ""
        return self._llm.detokenize([token_id], special=True).decode("utf-8", errors="ignore")

    def _render(self, messages: Sequence[Message]) -> tuple[list[int], list[str]]:
        response = self._format(messages=[m.as_dict() for m in messages])
        prompt: str = response.prompt
        stops = response.stop or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        tokens = list(self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
        if self._bos_id >= 0 and (not tokens or tokens[0] != self._bos_id):
            tokens.insert(0, self._bos_id)
        return tokens, stops

    # -------------------------------------------------------- TokenCounter
    @property
    def context_window(self) -> int:
        return self._cfg.context_window

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        return len(self._render(messages)[0])

    def count_text_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    # ------------------------------------------------------- ModelProvider
    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 256,
        temperature: float = 0.3,
        grammar: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        tokens, template_stops = self._render(messages)
        prompt_len = len(tokens)
        room = self.context_window - prompt_len
        if room < self.MIN_OUTPUT_TOKENS:
            raise ContextOverflowError(prompt_len, self.context_window)
        budget = min(max_output_tokens, room)

        kwargs: dict[str, Any] = {
            "prompt": tokens,
            "max_tokens": budget,
            "temperature": temperature,
            "stop": list(template_stops) + list(stop or []),
        }
        if grammar:
            kwargs["grammar"] = self._compiled(grammar)

        with self._lock:
            if self._cfg.reset_kv_between_calls:
                self._llm.reset()
            raw = self._llm.create_completion(**kwargs)

        choice = raw["choices"][0]
        usage = raw.get("usage") or {}
        return GenerationResult(
            text=choice.get("text", "").strip(),
            input_tokens=int(usage.get("prompt_tokens", prompt_len)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "stop",
            metadata={"output_budget": budget, "clamped": budget < max_output_tokens},
        )

    def _compiled(self, gbnf: str) -> Any:
        if gbnf not in self._grammar_cache:
            self._grammar_cache[gbnf] = self._grammar_compiler(gbnf)
        return self._grammar_cache[gbnf]

    def reset_context(self) -> None:
        """Drop the KV cache so the next call starts from an empty context."""
        with self._lock:
            self._llm.reset()

    # ---------------------------------------------------------------- vision
    def vision_status(self) -> tuple[bool, str]:
        handler = getattr(self._llm, "chat_handler", None)
        if handler is None or not hasattr(handler, "clip_model_path"):
            _, reason = vision_handler_for(self._cfg)
            return False, reason if reason != "ok" else "model loaded without a vision handler"
        return True, f"vision projector {handler.clip_model_path}"

    def describe_image(self, image_path, prompt: str, *, max_output_tokens: int = 300,
                       temperature: float = 0.2, system: str = "") -> GenerationResult:
        from .server_backend import image_data_uri

        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
            {"type": "text", "text": prompt},
        ]})
        with self._lock:
            raw = self._llm.create_chat_completion(messages=msgs, max_tokens=max_output_tokens, temperature=temperature)
        choice = raw["choices"][0]
        usage = raw.get("usage") or {}
        return GenerationResult(
            text=str(choice["message"].get("content") or "").strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    # -------------------------------------------------------------- lifecycle
    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is not None and hasattr(llm, "close"):
            llm.close()
        self._llm = None
        with LlamaCppEngine._class_lock:
            if LlamaCppEngine._live_instance is self:
                LlamaCppEngine._live_instance = None

    def __enter__(self) -> "LlamaCppEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
