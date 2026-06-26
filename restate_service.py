"""Restate ChatService — durable LLM execution layer.

This module is the ONLY place where the pydantic-ai agent actually runs.
FastAPI endpoints call this service via HTTP ingress; they never execute
the agent directly. This separation ensures:
  - Restate journals each LLM turn individually (RestateAgent with auto_wrap_tools)
  - pydantic-ai retry loops run outside the journal boundary → retries reach the model
  - AG-UI events are never emitted from here (replay would duplicate them on the client)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import restate
from graphiti_core import Graphiti
from opentelemetry import trace
from opentelemetry.propagate import extract as otel_extract
from pydantic_ai import Agent, RunContext
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessagesTypeAdapter
from restate import Context, ObjectContext, Service, VirtualObject
from restate.ext.pydantic import RestateAgent

import memory

tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Memory: one process-wide Graphiti client. Neo4j is client-server, so a single
# client (with its own connection pool) is shared across the FastAPI process and
# Restate handlers. Built lazily so module import never needs API keys; the
# FastAPI lifespan warms it at startup and closes it on shutdown.
# ---------------------------------------------------------------------------
_graphiti: Graphiti | None = None
_graphiti_lock = asyncio.Lock()


async def get_graphiti() -> Graphiti:
    """Return the process-wide Graphiti client, building + initialising it once."""
    global _graphiti
    async with _graphiti_lock:
        if _graphiti is None:
            client = memory.build_graphiti()
            await memory.init_graphiti(client)
            _graphiti = client
    return _graphiti


async def close_graphiti() -> None:
    """Close the shared Graphiti/Neo4j client. Call on app shutdown."""
    global _graphiti
    async with _graphiti_lock:
        if _graphiti is not None:
            await memory.close_graphiti(_graphiti)
            _graphiti = None


@dataclass
class MemoryDeps:
    """Injected into the agent so tools/instructions can reach memory."""

    graphiti: Graphiti
    user_id: str
    current_query: str
    pending_reflections: list[str]


# ---------------------------------------------------------------------------
# Base pydantic-ai agent — same model and system prompt as original chat_app.py
# ---------------------------------------------------------------------------
_base_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=MemoryDeps,
    tools=[tavily_search_tool(api_key=os.environ["TAVILY_API_KEY"], max_results=5)],
    system_prompt=(
        "You are a helpful personal assistant with web search and long-term memory. "
        "Use web search when you need current or real-time information. "
        "Use the search_memory tool to recall durable facts about the user when a "
        "question depends on personal context from past conversations."
    ),
    # Do NOT set retries=0 — RestateAgent journals each LLM turn individually,
    # so pydantic-ai retries (ModelRetry) still reach the model rather than
    # getting served the cached (failing) response from the journal.
)


@_base_agent.tool
async def search_memory(ctx: RunContext[MemoryDeps], query: str) -> str:
    """Search the user's long-term memory for durable facts relevant to a query.

    Use this to recall what is *currently* true about the user — their projects,
    preferences, opinions, and habits — especially when a question depends on
    personal context from earlier conversations. Superseded/stale facts are
    excluded, so the result reflects the present, not the past.
    """
    facts = await memory.search_memory(
        ctx.deps.graphiti,
        user_id=ctx.deps.user_id,
        query=query,
        durable_only=True,
    )
    if not facts:
        return "No relevant memory found."
    return "\n".join(f"- {fact}" for fact in facts)


@_base_agent.instructions
def inject_memory_context(ctx: RunContext[MemoryDeps]) -> str:
    """Surface proactively-detected changes (the nudge) on every run.

    MUST be `instructions`, not `system_prompt`: this app always passes
    message_history, and pydantic-ai does not re-send the system prompt once the
    history is non-empty — but instructions ARE re-applied every run. Using
    system_prompt here would make memory silently vanish on continuing turns.
    """
    if not ctx.deps.pending_reflections:
        return ""
    changes = "\n".join(f"- {change}" for change in ctx.deps.pending_reflections)
    return (
        "Since you last spoke with the user, these facts about them changed. "
        "If it is relevant to the conversation, gently and non-judgementally "
        "reflect the change back to them:\n" + changes
    )


# RestateAgent wraps the base agent to journal each LLM turn and tool call.
# auto_wrap_tools=True: Tavily HTTP calls are automatically wrapped in
# restate_context().run_typed(), preventing duplicate API calls on replay.
restate_agent = RestateAgent(_base_agent, auto_wrap_tools=True)

# ---------------------------------------------------------------------------
# Restate service definition
# ---------------------------------------------------------------------------
chat_service = Service("ChatService")


@chat_service.handler()
async def run(ctx: Context, payload: dict) -> dict:
    """Run the pydantic-ai agent durably inside a Restate handler.

    Payload keys:
      prompt          (str)  — user message
      message_history (list) — JSON-serialised ModelMessages, may be empty
      otel_context    (dict) — W3C traceparent carrier for span linking

    Returns:
      output          (str)  — agent text response
      new_messages    (list) — JSON-serialised new ModelMessages for this turn
      timestamp       (str)  — ISO-8601 UTC completion time
    """
    prompt: str = payload["prompt"]
    user_id: str = payload.get("user_id", "default")
    # The user's display NAME (not the id). Graphiti needs a named speaker to mint a
    # Person node, otherwise first-person self-state ("I feel…") forms no edges.
    user_name: str = payload.get("user_name", "User")
    history_json: list = payload.get("message_history", [])
    message_history = (
        ModelMessagesTypeAdapter.validate_python(history_json) if history_json else []
    )

    # Extract OTEL trace context injected by the FastAPI layer.
    # We pass it via the JSON payload rather than relying on Restate header
    # forwarding, which is undocumented for the Python SDK.
    otel_ctx = otel_extract(payload.get("otel_context", {}))

    # Read + clear proactively-detected changes for this user (the nudge surface).
    # drain_reflections is an exclusive Virtual Object handler, so it serializes
    # with ingest() — no read/write race on the per-user reflection state.
    pending_reflections: list[str] = await ctx.object_call(
        drain_reflections, key=user_id, arg={}
    )

    graphiti = await get_graphiti()
    deps = MemoryDeps(
        graphiti=graphiti,
        user_id=user_id,
        current_query=prompt,
        pending_reflections=pending_reflections,
    )

    with tracer.start_as_current_span(
        "restate.ChatService.run",
        context=otel_ctx,
        attributes={
            # restate.invocation_id is captured at the FastAPI layer from the
            # x-restate-id response header; it is not accessible from ctx here.
            "restate.service": "ChatService",
            "prompt.length": len(prompt),
        },
    ) as span:
        # RestateAgent.run() does NOT take ctx as first argument.
        # The Restate context is captured implicitly via an async context var —
        # the same mechanism that makes restate_context() work inside tools.
        result = await restate_agent.run(
            prompt,
            message_history=message_history,
            deps=deps,
        )
        span.set_attribute("output.length", len(str(result.output)))

    # Fire-and-forget durable ingestion of this turn, ORDERED per user.
    # object_send to a Virtual Object queues serially per key, which is what
    # Graphiti's sequential contradiction detection requires. It is journaled,
    # so it neither blocks the response nor double-runs on replay.
    # Admission gate: skip trivial turns so we don't pay ~7 extraction LLM calls
    # (and pollute the graph) on a "thanks"/"ok".
    if memory.should_ingest(prompt):
        ctx.object_send(
            ingest,
            key=user_id,
            arg={
                # Frame the turn with the user's NAME as the speaker so Graphiti
                # extracts them as a Person entity and self-state becomes edges.
                "episode_body": f"{user_name}: {prompt}\nAssistant: {result.output}",
                "reference_time": datetime.now(tz=UTC).isoformat(),
            },
        )

    return {
        "output": result.output,
        "new_messages": ModelMessagesTypeAdapter.dump_python(
            result.new_messages(), mode="json"
        ),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# MemoryService — a Virtual Object keyed by user_id.
#
# Why a Virtual Object (not a plain Service): Restate runs at most one exclusive
# handler per key at a time, in arrival order — a durable per-user queue. Graphiti's
# add_episode() performs edge invalidation that depends on prior graph state, so
# ingestion MUST be serialized per user. The object's K/V state also chains episodes
# (last_episode_uuid) and holds pending_reflections for the proactive nudge.
# ---------------------------------------------------------------------------
memory_service = VirtualObject("MemoryService")


@memory_service.handler()
async def ingest(ctx: ObjectContext, payload: dict) -> dict:
    """Ingest one chat turn into the user's temporal graph (durable, serialized)."""
    last_episode_uuid = await ctx.get("last_episode_uuid", type_hint=str)
    episode_body: str = payload["episode_body"]
    reference_time = datetime.fromisoformat(payload["reference_time"])

    async def _do() -> dict:
        graphiti = await get_graphiti()
        episode_uuid, reflections = await memory.ingest_turn(
            graphiti,
            user_id=ctx.key(),
            episode_body=episode_body,
            reference_time=reference_time,
            previous_episode_uuid=last_episode_uuid,
        )
        return {"episode_uuid": episode_uuid, "reflections": reflections}

    # Journal the extraction result once; on replay it is NOT re-run (no duplicate
    # LLM calls / writes), it is replayed from the journal.
    result = await ctx.run_typed("ingest_episode", _do)

    ctx.set("last_episode_uuid", result["episode_uuid"])
    pending = await ctx.get("pending_reflections", type_hint=list) or []
    ctx.set("pending_reflections", pending + result["reflections"])
    return result


@memory_service.handler()
async def drain_reflections(ctx: ObjectContext, _arg: dict | None = None) -> list:
    """Read and clear pending reflections for this user (exclusive → no race)."""
    notes = await ctx.get("pending_reflections", type_hint=list) or []
    if notes:
        ctx.clear("pending_reflections")
    return notes


# ---------------------------------------------------------------------------
# ASGI app — Restate calls this for service discovery and handler invocations.
# Mounted at /restate-service inside FastAPI (at module level, not in lifespan).
# ---------------------------------------------------------------------------
restate_asgi = restate.app(services=[chat_service, memory_service])

# ---------------------------------------------------------------------------
# Production pattern for durable Tavily (commented — not used in this PoC).
#
# The built-in tavily_search_tool() already achieves idempotency in the PoC
# via auto_wrap_tools=True on RestateAgent. The explicit pattern below is
# equivalent and shows how to apply it to any external HTTP call in a tool:
#
# from restate import restate_context
# from tavily import AsyncTavilyClient
#
# @_base_agent.tool_plain
# async def durable_tavily_search(query: str) -> str:
#     client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
#     async def _search():
#         r = await client.search(query, max_results=5)
#         return "\n".join(x["content"] for x in r["results"])
#     return await restate_context().run_typed("tavily", _search)
# ---------------------------------------------------------------------------
