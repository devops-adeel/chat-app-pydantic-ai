"""Personal chat assistant with web search, built with FastAPI and pydantic-ai.

Architecture: FastAPI is a stateless streaming layer. All agent execution happens
inside the Restate ChatService (restate_service.py). FastAPI calls Restate via HTTP
and streams the completed result back to the browser.

Run with:

    docker compose up restate -d
    uv run uvicorn chat_app:app --reload
"""

from __future__ import annotations as _annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import fastapi
import httpx
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from opentelemetry import trace
from opentelemetry.propagate import inject as otel_inject
from pydantic_ai import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UnexpectedModelBehavior,
    UserPromptPart,
)
from pydantic_ai.messages import ModelMessagesTypeAdapter
from typing_extensions import TypedDict

from instrumentation import configure_telemetry
from restate_service import close_graphiti, get_graphiti, restate_asgi

load_dotenv()

# Must call before any Agent instances are created (instrument_all hooks into Agent.__init__)
configure_telemetry()

tracer = trace.get_tracer(__name__)

RESTATE_INGRESS = os.environ.get("RESTATE_INGRESS", "http://localhost:8080")
RESTATE_ADMIN = os.environ.get("RESTATE_ADMIN", "http://localhost:9070")
# On Mac Docker Desktop, host.docker.internal resolves to the host machine.
# On Linux Docker: set SERVICE_ENDPOINT=http://172.17.0.1:8000/restate-service
SERVICE_ENDPOINT = os.environ.get(
    "SERVICE_ENDPOINT", "http://host.docker.internal:8000/restate-service"
)
# Single-user personal assistant: one stable id scopes the user's memory graph
# (Restate Virtual Object key + Graphiti group_id).
MEMORY_USER_ID = os.environ.get("MEMORY_USER_ID", "default")

THIS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Restate ASGI sub-app mounted at module level — must be ready before the
# first request arrives (not in lifespan, which fires after binding).
# ---------------------------------------------------------------------------
app = fastapi.FastAPI()
app.mount("/restate-service", restate_asgi)


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    # Register this FastAPI app as a Restate deployment.
    # Restate will call /restate-service to discover ChatService and invoke handlers.
    # Re-registration on restart is safe (idempotent).
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{RESTATE_ADMIN}/deployments",
                json={"uri": SERVICE_ENDPOINT},
            )
            resp.raise_for_status()
            print(f"[restate] registered deployment: {SERVICE_ENDPOINT}")
    except Exception as exc:
        # Non-fatal — register manually if needed:
        # curl -X POST localhost:9070/deployments -H 'Content-Type: application/json' \
        #      -d '{"uri": "http://host.docker.internal:8000/restate-service"}'
        print(f"[restate] registration skipped (start Restate first): {exc}")

    # Warm the Graphiti/Neo4j client once: build indices + open the connection pool
    # so the first chat turn doesn't pay that cost (and to avoid a startup race on
    # index creation). Non-fatal if Neo4j isn't up yet — falls back to lazy init.
    try:
        await get_graphiti()
        print("[memory] Graphiti client ready (Neo4j indices built)")
    except Exception as exc:
        print(f"[memory] Graphiti warm-up skipped (start Neo4j first): {exc}")

    yield {"messages": []}

    # Shutdown: close the Neo4j connection pool.
    await close_graphiti()


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# Message types and helpers (unchanged from original)
# ---------------------------------------------------------------------------


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def is_displayable(m: ModelMessage) -> bool:
    if isinstance(m, ModelRequest):
        return bool(m.parts) and isinstance(m.parts[0], UserPromptPart)
    if isinstance(m, ModelResponse):
        return bool(m.parts) and isinstance(m.parts[0], TextPart)
    return False


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                "role": "user",
                "timestamp": first_part.timestamp.isoformat(),
                "content": first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                "role": "model",
                "timestamp": m.timestamp.isoformat(),
                "content": first_part.content,
            }
    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


async def get_messages(request: Request) -> list[ModelMessage]:
    return request.state.messages


# ---------------------------------------------------------------------------
# Shared Restate call helper
# ---------------------------------------------------------------------------


async def _call_restate(prompt: str, messages: list[ModelMessage]) -> tuple[dict, str]:
    """Invoke ChatService.run() via Restate HTTP ingress.

    Blocks until Restate completes the full agent run (including all tool calls).
    Returns (result_dict, restate_invocation_id).

    The OTEL trace context is injected into the payload (not headers) because
    Restate's W3C traceparent forwarding to Python handlers is undocumented.
    """
    otel_carrier: dict[str, str] = {}
    otel_inject(otel_carrier)

    payload = {
        "prompt": prompt,
        "user_id": MEMORY_USER_ID,
        "message_history": ModelMessagesTypeAdapter.dump_python(messages, mode="json"),
        "otel_context": otel_carrier,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{RESTATE_INGRESS}/ChatService/run",
            json=payload,
        )
        resp.raise_for_status()
        invocation_id = resp.headers.get("x-restate-id", "unknown")
        return resp.json(), invocation_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / "chat_app.html"), media_type="text/html")


@app.get("/chat_app.ts")
async def main_ts() -> FileResponse:
    return FileResponse((THIS_DIR / "chat_app.ts"), media_type="text/plain")


@app.get("/chat/")
async def get_chat(messages: list[ModelMessage] = Depends(get_messages)) -> Response:
    displayable = [m for m in messages if is_displayable(m)]
    return Response(
        b"\n".join(json.dumps(to_chat_message(m)).encode("utf-8") for m in displayable),
        media_type="text/plain",
    )


@app.post("/chat/")
async def post_chat(
    prompt: Annotated[str, fastapi.Form()],
    messages: list[ModelMessage] = Depends(get_messages),
) -> StreamingResponse:
    """SSE endpoint — stateless streaming layer.

    The agent runs inside Restate (ChatService.run). This endpoint:
      1. Yields the user message immediately so the UI renders it without waiting.
      2. Calls Restate and blocks until the full agent run completes.
      3. Updates in-memory message history with the journaled result.
      4. Yields the completed model response as a single NDJSON frame.

    UX trade-off: no token-by-token streaming (Restate handlers are synchronous).
    The user sees a loading state until Restate finishes, then the full response.
    Future: use Restate awakeables to stream paragraph-by-paragraph.
    """

    async def stream_messages():
        # Step 1: yield user message immediately — no agent call in this layer
        yield (
            json.dumps(
                {
                    "role": "user",
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "content": prompt,
                }
            ).encode("utf-8")
            + b"\n"
        )

        # Step 2: call Restate durably — blocks until agent run completes
        with tracer.start_as_current_span(
            "fastapi.chat.post", attributes={"prompt.length": len(prompt)}
        ):
            result, invocation_id = await _call_restate(prompt, messages)

        # Step 3: update in-memory history with the durably journaled result
        new_msgs = ModelMessagesTypeAdapter.validate_python(result["new_messages"])
        messages.extend(new_msgs)

        # Step 4: yield completed response (single NDJSON frame)
        yield (
            json.dumps(
                {
                    "role": "model",
                    "timestamp": result["timestamp"],
                    "content": result["output"],
                    # restate_invocation_id links this response to the Restate journal
                    # and (when live) to the Langfuse trace. Visible in browser devtools.
                    "restate_invocation_id": invocation_id,
                }
            ).encode("utf-8")
            + b"\n"
        )

    return StreamingResponse(stream_messages(), media_type="text/plain")


@app.post("/pydantic-agent")
async def run_ag_ui(request: Request) -> StreamingResponse:
    """AG-UI endpoint — bridges Restate for durable execution.

    AG-UI events are emitted HERE (FastAPI layer), never inside the Restate handler.

    Why: Restate replays handlers on failure. If a handler emitted 50 TEXT_MESSAGE_CONTENT
    events before crashing, replay would re-emit all 50 — the client would see duplicates.
    By completing the Restate invocation first and emitting events from FastAPI, the SSE
    connection is independent of Restate's journal.

    AGUIAdapter.dispatch_request() is intentionally NOT used because:
    - It calls agent.run_stream_events() internally
    - RestateAgent.run() requires a Restate execution context (available only inside handlers)
    - Calling RestateAgent from FastAPI context would raise a missing-context error
    """
    body = await request.json()
    ag_messages = body.get("messages", [])
    user_message = next(
        (m["content"] for m in reversed(ag_messages) if m.get("role") == "user"),
        "",
    )
    thread_id: str = body.get("threadId", str(uuid.uuid4()))
    run_id = str(uuid.uuid4())
    msg_id = str(uuid.uuid4())

    # Call Restate — complete the agent run durably before emitting any AG-UI events
    result, invocation_id = await _call_restate(user_message, [])
    text: str = result["output"]

    async def emit_ag_ui_events():
        def sse(event_type: str, data: dict) -> bytes:
            return f"data: {json.dumps({'type': event_type, **data})}\n\n".encode()

        yield sse("RUN_STARTED", {"threadId": thread_id, "runId": run_id})
        yield sse("TEXT_MESSAGE_START", {"messageId": msg_id, "role": "assistant"})

        # Chunk the completed text for a streaming-like UX in the AG-UI frontend
        chunk_size = 30
        for i in range(0, len(text), chunk_size):
            yield sse(
                "TEXT_MESSAGE_CONTENT",
                {"messageId": msg_id, "delta": text[i : i + chunk_size]},
            )

        yield sse("TEXT_MESSAGE_END", {"messageId": msg_id})
        yield sse(
            "RUN_FINISHED",
            {
                "threadId": thread_id,
                "runId": run_id,
                "restateInvocationId": invocation_id,
            },
        )

    return StreamingResponse(emit_ag_ui_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chat_app:app", reload=True, reload_dirs=[str(THIS_DIR)])
