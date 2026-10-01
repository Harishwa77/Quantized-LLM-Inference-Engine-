"""Optional FastAPI application exposing the same routes as :mod:`llmopt.serving.app`.

FastAPI is not a hard dependency. Import this module only when you want
automatic request validation, OpenAPI docs, or an ASGI deployment::

    pip install "llmopt[serving]"

    import uvicorn
    from llmopt.serving.fastapi_app import create_app

    uvicorn.run(create_app(engine), host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llmopt.config import ServerConfig
from llmopt.engine.llm_engine import LLMEngine
from llmopt.utils.logging import get_logger

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised only without fastapi
    raise ImportError(
        "llmopt.serving.fastapi_app requires fastapi. "
        "Install it with: pip install 'llmopt[serving]'"
    ) from exc

__all__ = ["create_app"]

logger = get_logger("serving.fastapi")


def create_app(engine: LLMEngine, config: ServerConfig | None = None) -> FastAPI:
    """Build a FastAPI app bound to ``engine``.

    Args:
        engine: Engine that serves requests.
        config: Server settings; defaults to :class:`ServerConfig`.

    Returns:
        A configured ``FastAPI`` instance.
    """
    from llmopt.serving.app import OpenAIServer

    cfg = config or ServerConfig()
    app_server = OpenAIServer(engine, cfg)
    api = FastAPI(
        title="llmopt",
        version="0.1.0",
        description="OpenAI-compatible API backed by a custom paged-attention engine.",
    )

    def check_auth(authorization: str | None = Header(default=None)) -> None:
        if not app_server.authorize(authorization):
            raise HTTPException(status_code=401, detail="invalid api key")

    auth = [Depends(check_auth)]

    @api.get("/health", dependencies=auth)
    def health() -> dict[str, Any]:
        """Liveness probe."""
        return app_server.get_health()

    @api.get("/info", dependencies=auth)
    def info() -> dict[str, Any]:
        """Engine configuration, statistics, and memory estimate."""
        return app_server.get_info()

    @api.get("/metrics", response_class=PlainTextResponse, dependencies=auth)
    def metrics() -> str:
        """Prometheus text exposition."""
        return app_server.get_metrics()

    @api.get("/v1/models", dependencies=auth)
    def models() -> dict[str, Any]:
        """List served models."""
        return app_server.list_models()

    @api.post("/v1/completions", dependencies=auth)
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        """Text completion endpoint."""
        try:
            return app_server.create_completion(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/v1/chat/completions", dependencies=auth)
    def chat_completions(payload: dict[str, Any]):
        """Chat completion endpoint with optional SSE streaming."""
        try:
            result = app_server.create_chat_completion(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result.get("__stream__"):
            return result
        result.pop("__stream__", None)
        events = result.pop("_events")
        return StreamingResponse(
            _sse(events), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @api.exception_handler(Exception)
    def unhandled(_: Any, exc: Exception) -> JSONResponse:
        logger.exception("unhandled server error")
        return JSONResponse(status_code=500, content={"error": {"message": str(exc)}})

    logger.info("fastapi app ready for %s", cfg.model_id)
    return api


async def _sse(events: Any) -> AsyncIterator[bytes]:
    for event in events:
        yield event
