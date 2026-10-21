"""ModelProvider + TokenCounter + vision over llama.cpp's `llama-server`.

Why a second backend: the server is llama.cpp's reference implementation, so
new model families (and their vision projectors) work there first. The rest
of Tessera can't tell the difference — this satisfies the same protocols.

Exact token accounting uses the server's own endpoints:
  POST /apply-template  → the rendered prompt (model's chat template)
  POST /tokenize        → its tokens (parse_special so control tokens count right)
  POST /completion      → generation from those exact token ids
Endpoint and field names follow tools/server/README.md in llama.cpp.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from ..core.types import GenerationResult, Message

_FINISH = {"eos": "stop", "word": "stop", "limit": "length", "none": "length"}


class ServerError(RuntimeError):
    pass


class LlamaServerEngine:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", *, timeout_s: float = 300.0, slot: int = 0,
                 context_window: int | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._slot = slot
        self._lock = threading.Lock()
        self._n_ctx = context_window or int(self._get("/props")["default_generation_settings"]["n_ctx"])

    # --------------------------------------------------------------- http
    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ServerError(f"{path}: HTTP {exc.code} {exc.read()[:300]!r}") from exc
        except urllib.error.URLError as exc:
            raise ServerError(f"cannot reach llama-server at {self._base}: {exc.reason}") from exc

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> Any:
        return self._request("POST", path, body)

    # -------------------------------------------------------- TokenCounter
    @property
    def context_window(self) -> int:
        return self._n_ctx

    def _tokens(self, messages: Sequence[Message]) -> list[int]:
        prompt = self._post("/apply-template", {"messages": [m.as_dict() for m in messages]})["prompt"]
        return list(self._post("/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"])

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        return len(self._tokens(messages))

    # ------------------------------------------------------- ModelProvider
    def generate(self, messages: Sequence[Message], *, max_output_tokens: int = 256, temperature: float = 0.3,
                 grammar: str | None = None, stop: Sequence[str] | None = None) -> GenerationResult:
        tokens = self._tokens(messages)
        room = self._n_ctx - len(tokens)
        if room < 1:
            from .engine import ContextOverflowError
            raise ContextOverflowError(len(tokens), self._n_ctx)
        budget = min(max_output_tokens, room)
        body: dict[str, Any] = {
            "prompt": tokens, "n_predict": budget, "temperature": temperature,
            "cache_prompt": True, "id_slot": self._slot,
        }
        if grammar:
            body["grammar"] = grammar
        if stop:
            body["stop"] = list(stop)
        with self._lock:
            r = self._post("/completion", body)
        predicted = r.get("tokens_predicted", (r.get("timings") or {}).get("predicted_n", 0))
        return GenerationResult(
            text=str(r.get("content", "")).strip(),
            input_tokens=int(r.get("tokens_evaluated", len(tokens))),
            output_tokens=int(predicted),
            finish_reason=_FINISH.get(r.get("stop_type", "eos"), "stop"),
            metadata={"output_budget": budget, "tokens_cached": r.get("tokens_cached"), "truncated": r.get("truncated")},
        )

    def reset_context(self) -> None:
        with self._lock:
            self._post(f"/slots/{self._slot}?action=erase", {})

    # ---------------------------------------------------------------- vision
    def describe_image(self, image_path: str | Path, prompt: str, *, max_output_tokens: int = 300,
                       temperature: float = 0.2, system: str = "") -> GenerationResult:
        url = image_data_uri(image_path)
        msgs: list[dict] = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ]})
        with self._lock:
            r = self._post("/v1/chat/completions", {"messages": msgs, "max_tokens": max_output_tokens,
                                                    "temperature": temperature})
        choice = r["choices"][0]
        usage = r.get("usage") or {}
        return GenerationResult(
            text=str(choice["message"].get("content") or "").strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def vision_status(self) -> tuple[bool, str]:
        try:
            models = self._get("/v1/models")
        except ServerError as exc:
            return False, str(exc)
        text = json.dumps(models).lower()
        if "multimodal" in text or "vision" in text:
            return True, "server reports multimodal capability"
        return False, "server does not report multimodal capability (start llama-server with --mmproj)"

    def close(self) -> None:
        pass


def image_data_uri(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
