"""OpenAI-compatible HTTP server for :class:`~llmopt.engine.llm_engine.LLMEngine`.

Uses only the standard library so the server has no hard dependency on a web
framework. ``fastapi`` is used instead when installed (see
:mod:`llmopt.serving.fastapi_app`), which adds validation and async support.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llmopt.config import ServerConfig
from llmopt.engine.llm_engine import LLMEngine
from llmopt.engine.request import SamplingParams
from llmopt.utils.logging import get_logger
from llmopt.utils.misc import atomic_write_json, human_bytes

__all__ = ["OpenAIServer", "create_server", "serve"]

logger = get_logger("serving.http")

_DEFAULT_MODELS = ("llmopt-tiny", "llmopt-small")


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages into the plain-text prompt this engine expects."""
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            # Accept the multimodal shape by concatenating text parts.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def _usage(engine: LLMEngine, prompt_ids: list[int], output_ids: list[int]) -> dict[str, int]:
    return {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(output_ids),
        "total_tokens": len(prompt_ids) + len(output_ids),
    }


def _estimate_memory(engine: LLMEngine) -> dict[str, Any]:
    from llmopt.utils.misc import estimate_kv_cache_bytes

    config = engine.config.model
    cache = engine.config.cache
    total = estimate_kv_cache_bytes(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        cache.num_gpu_blocks,
        cache.block_size,
        itemsize=engine.cache.key_cache.element_size(),
    )
    return {
        "kv_cache_bytes": total,
        "kv_cache_human": human_bytes(total),
        "weights_bytes": engine.model.parameter_bytes(),
        "weights_human": human_bytes(engine.model.parameter_bytes()),
        "blocks": cache.num_gpu_blocks,
        "block_size": cache.block_size,
        "max_concurrent_tokens": cache.num_gpu_blocks * cache.block_size,
    }


class OpenAIServer:
    """Request handler translating OpenAI payloads into engine calls.

    Args:
        engine: The engine that serves requests.
        config: HTTP settings.
    """

    def __init__(self, engine: LLMEngine, config: ServerConfig | None = None) -> None:
        self.engine = engine
        self.config = config or ServerConfig()
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._num_requests = 0
        self._num_completions = 0

    # ------------------------------------------------------------------ auth

    def authorize(self, header: str | None) -> bool:
        """Check the bearer token when an API key is configured."""
        if not self.config.api_key:
            return True
        expected = f"Bearer {self.config.api_key}"
        return header is not None and header == expected

    # ---------------------------------------------------------------- routes

    def list_models(self) -> dict[str, Any]:
        """``GET /v1/models``"""
        model = self.config.model_id
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(self.started_at),
                    "owned_by": "llmopt",
                }
            ],
        }

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """``GET /v1/models/{id}``"""
        if model_id not in (self.config.model_id, *_DEFAULT_MODELS):
            return None
        return {
            "id": model_id,
            "object": "model",
            "created": int(self.started_at),
            "owned_by": "llmopt",
        }

    def create_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /v1/completions``"""
        self._tick()
        prompt = payload.get("prompt", "")
        prompts = prompt if isinstance(prompt, list) else [prompt]
        params = SamplingParams.from_openai(payload)
        stream = bool(payload.get("stream", False))
        if stream:
            raise ValueError("use /v1/chat/completions for streaming, or set stream=false")

        completions = []
        for index, single in enumerate(prompts):
            result = self.engine.generate_sync(str(single), params)
            completions.append(
                {
                    "index": index,
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "finish_reason": result.finish_reason.value if result.finish_reason else None,
                }
            )
        prompt_ids = self.engine.tokenizer.encode(str(prompts[0]))
        completion_ids = completions[0]["token_ids"]
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": payload.get("model", self.config.model_id),
            "choices": completions,
            "usage": _usage(self.engine, prompt_ids, completion_ids),
        }

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /v1/chat/completions``"""
        self._tick()
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("'messages' must be a non-empty array")
        params = SamplingParams.from_openai(payload)
        prompt = _messages_to_prompt(messages)
        prompt_ids = self.engine.tokenizer.encode(prompt)
        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if payload.get("stream"):
            return {
                "__stream__": True,
                "id": request_id,
                "created": created,
                "model": payload.get("model", self.config.model_id),
                "prompt_tokens": len(prompt_ids),
                "_events": self._stream_events(prompt, params, request_id, created, prompt_ids),
            }

        result = self.engine.generate_sync(prompt, params)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": payload.get("model", self.config.model_id),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason.value if result.finish_reason else None,
                }
            ],
            "usage": _usage(self.engine, prompt_ids, result.token_ids),
            "llmopt_metrics": result.metrics,
        }

    def _stream_events(
        self,
        prompt: str,
        params: SamplingParams,
        request_id: str,
        created: int,
        prompt_ids: list[int],
    ) -> Iterator[bytes]:
        """Yield SSE frames for a streaming chat completion."""
        model = self.config.model_id

        def frame(delta: dict[str, Any], finish: str | None) -> bytes:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(chunk)}\n\n".encode()

        yield frame({"role": "assistant", "content": ""}, None)
        final_finish = "stop"
        final_tokens: list[int] = []
        for update in self.engine.stream(prompt, params):
            if update.done:
                final_tokens = update.token_ids
                final_finish = update.finish_reason.value if update.finish_reason else "stop"
                break
            if update.delta:
                yield frame({"content": update.delta}, None)
        yield frame({}, final_finish)
        yield (
            "data: "
            + json.dumps(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": _usage(self.engine, prompt_ids, final_tokens),
                }
            )
            + "\n\n"
        ).encode()
        yield b"data: [DONE]\n\n"

    def get_metrics(self) -> str:
        """``GET /metrics`` in Prometheus text format."""
        lines = [
            "# HELP llmopt_uptime_seconds Seconds since the server started.",
            "# TYPE llmopt_uptime_seconds gauge",
            f"llmopt_uptime_seconds {time.time() - self.started_at:.3f}",
            "# HELP llmopt_http_requests_total HTTP requests handled.",
            "# TYPE llmopt_http_requests_total counter",
            f"llmopt_http_requests_total {self._num_requests}",
        ]
        return "\n".join(lines) + "\n" + self.engine.metrics_text()

    def get_health(self) -> dict[str, Any]:
        """``GET /health``"""
        return {
            "status": "ok",
            "model": self.config.model_id,
            "uptime_s": round(time.time() - self.started_at, 3),
            "device": str(self.engine.device),
            "dtype": str(self.engine.dtype),
        }

    def get_info(self) -> dict[str, Any]:
        """``GET /info`` -- full engine configuration and statistics."""
        return {
            "config": self.engine.config.to_dict(),
            "stats": self.engine.stats(),
            "memory": _estimate_memory(self.engine),
        }

    def snapshot(self, path: str) -> None:
        """Persist ``/info`` to ``path`` as JSON."""
        atomic_write_json(path, self.get_info())

    def _tick(self) -> None:
        with self._lock:
            self._num_requests += 1


def _make_handler(server: OpenAIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llmopt/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            if server.config.log_requests:
                logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode() or "{}")

        def _route(self, method: str) -> bool:
            path = self.path.split("?")[0].rstrip("/") or "/"
            if not server.authorize(self.headers.get("Authorization")):
                self._json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "invalid api key"}})
                return True
            try:
                if method == "GET":
                    if path in ("/health", "/v1/health"):
                        self._json(HTTPStatus.OK, server.get_health())
                    elif path == "/info":
                        self._json(HTTPStatus.OK, server.get_info())
                    elif path == "/v1/models":
                        self._json(HTTPStatus.OK, server.list_models())
                    elif path.startswith("/v1/models/"):
                        model = server.get_model(path.rsplit("/", 1)[-1])
                        if model is None:
                            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "no such model"}})
                        else:
                            self._json(HTTPStatus.OK, model)
                    elif path == "/metrics":
                        self._send(HTTPStatus.OK, server.get_metrics().encode(), "text/plain; version=0.0.4")
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": {"message": f"no route for {path}"}})
                elif method == "POST":
                    payload = self._read_json()
                    if path == "/v1/completions":
                        self._json(HTTPStatus.OK, server.create_completion(payload))
                    elif path == "/v1/chat/completions":
                        result = server.create_chat_completion(payload)
                        if result.get("__stream__"):
                            self._stream(result)
                        else:
                            self._json(HTTPStatus.OK, result)
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": {"message": f"no route for {path}"}})
                else:
                    self._json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": {"message": method}})
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            except Exception as exc:  # noqa: BLE001 - surface errors as JSON, not crashes
                logger.exception("request failed")
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc)}})
            return True

        def _stream(self, payload: dict[str, Any]) -> None:
            events = payload.pop("_events")
            payload.pop("__stream__", None)
            first = json.dumps(payload).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._write_chunk(first + b"\n\n")
            for event in events:
                self._write_chunk(event)
            self._write_chunk(b"")

        def _write_chunk(self, payload: bytes) -> None:
            self.wfile.write(f"{len(payload):X}\r\n".encode())
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            self._route("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._route("POST")

    return Handler


def create_server(
    engine: LLMEngine, config: ServerConfig | None = None
) -> tuple[ThreadingHTTPServer, OpenAIServer]:
    """Build (but do not start) an HTTP server for ``engine``."""
    app = OpenAIServer(engine, config)
    httpd = ThreadingHTTPServer((app.config.host, app.config.port), _make_handler(app))
    return httpd, app


def serve(
    engine: LLMEngine,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_id: str = "llmopt-tiny",
    api_key: str | None = None,
    block: bool = True,
    config: ServerConfig | None = None,
) -> ThreadingHTTPServer | None:
    """Run the server until interrupted.

    Args:
        engine: Engine to serve.
        host: Bind address.
        port: Bind port.
        model_id: Model name reported by ``/v1/models``.
        api_key: Optional bearer token required on every request.
        block: Block the calling thread. Set ``False`` to run in the background.
        config: Full config; overrides the individual arguments when given.

    Returns:
        The running server when ``block`` is ``False``, otherwise ``None``.
    """
    if config is None:
        config = ServerConfig(host=host, port=port, model_id=model_id, api_key=api_key)
    httpd, app = create_server(engine, config)
    logger.info("serving %s on http://%s:%d", config.model_id, config.host, config.port)
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.info("shutting down")
    finally:
        httpd.server_close()
        if config.snapshot_path:
            app.snapshot(config.snapshot_path)
            logger.info("wrote snapshot to %s", config.snapshot_path)
    return None
