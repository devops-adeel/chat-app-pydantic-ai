"""Wiring + temporal-behaviour tests for the Graphiti memory layer.

These are deterministic and need no API keys or database: the model is
TestModel/FunctionModel and `memory.search_memory` is monkeypatched. They verify
the *wiring* and the transient/durable contract — not Graphiti's extraction quality
(that needs the live harness in `eval_memory.py` against a real graph).
"""

from __future__ import annotations

import os
import types

# restate_service builds an Agent + Tavily tool at import — give it dummy creds.
os.environ.setdefault("TAVILY_API_KEY", "dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("NEO4J_PASSWORD", "dummy")

from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

import memory  # noqa: E402
import restate_service as rs  # noqa: E402


def make_deps(graphiti: object | None = None, pending: list[str] | None = None):
    return rs.MemoryDeps(
        graphiti=graphiti,
        user_id="test-user",
        current_query="q",
        pending_reflections=pending or [],
    )


# --- Ontology invariants ---------------------------------------------------
def test_transient_and_durable_partition_edge_types():
    transient, durable = (
        set(memory.TRANSIENT_EDGE_TYPES),
        set(memory.DURABLE_EDGE_TYPES),
    )
    assert transient.isdisjoint(durable)
    assert transient | durable == set(memory.EDGE_TYPES)


# --- Admission gate --------------------------------------------------------
def test_should_ingest_skips_trivial_turns():
    for trivial in ["thanks", "ok", "Okay!", "  yep ", "lol", "k", "👍👍"[:1]]:
        assert memory.should_ingest(trivial) is False


def test_should_ingest_keeps_substantive_and_terse_important_turns():
    # Terse-but-important facts MUST pass (fidelity over cost).
    assert memory.should_ingest("I quit my job") is True
    assert memory.should_ingest("I've shelved the thesis to focus on the new role.")
    assert memory.should_ingest("low energy today, autistic burnout") is True


# --- Reflection extraction (pure) ------------------------------------------
def test_reflections_only_from_invalidated_edges():
    edges = [
        types.SimpleNamespace(fact="works on thesis", invalid_at=None, expired_at=None),
        types.SimpleNamespace(
            fact="works on new job", invalid_at=None, expired_at=None
        ),
        types.SimpleNamespace(
            fact="focus is the thesis",
            invalid_at="2026-06-26T00:00:00Z",
            expired_at=None,
        ),
    ]
    assert memory._reflections_from_edges(edges) == ["changed: focus is the thesis"]


# --- Instruction injection = the proactive nudge ---------------------------
def test_inject_memory_context_surfaces_reflections():
    ctx = types.SimpleNamespace(
        deps=make_deps(pending=["changed: focus is now the job"])
    )
    assert "focus is now the job" in rs.inject_memory_context(ctx)


def test_inject_memory_context_empty_when_no_reflections():
    ctx = types.SimpleNamespace(deps=make_deps(pending=[]))
    assert rs.inject_memory_context(ctx) == ""


# --- Tool registration (wiring) --------------------------------------------
async def test_search_memory_tool_is_registered(monkeypatch):
    async def fake_search(*args, **kwargs):
        return []

    monkeypatch.setattr(memory, "search_memory", fake_search)
    # call_tools restricts TestModel to search_memory so the real Tavily tool
    # (which would hit the network with a dummy key) is never invoked.
    model = TestModel(call_tools=["search_memory"])
    with rs._base_agent.override(model=model):
        await rs._base_agent.run("hello", deps=make_deps())

    tool_names = [t.name for t in model.last_model_request_parameters.function_tools]
    assert "search_memory" in tool_names
    assert any("tavily" in name for name in tool_names)


# --- Temporal behaviour: current-not-stale (the whole point) ---------------
async def test_agent_surfaces_current_fact_not_stale(monkeypatch):
    async def fake_search(
        graphiti, *, user_id, query, durable_only=True, num_results=8
    ):
        # Retrieval must exclude transient mood/energy from durable belief.
        assert durable_only is True
        return [
            "The user's current main focus is their new job; the thesis is shelved."
        ]

    monkeypatch.setattr(memory, "search_memory", fake_search)

    calls = {"n": 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="search_memory",
                        args={"query": "what is the user's current focus?"},
                    )
                ]
            )
        tool_text = ""
        for msg in messages:
            for part in getattr(msg, "parts", []):
                if getattr(part, "tool_name", None) == "search_memory" and hasattr(
                    part, "content"
                ):
                    tool_text = str(part.content)
        return ModelResponse(
            parts=[TextPart(content=f"Your current focus: {tool_text}")]
        )

    with rs._base_agent.override(model=FunctionModel(model_fn)):
        result = await rs._base_agent.run(
            "What am I focused on now?", deps=make_deps(graphiti=object())
        )

    out = result.output.lower()
    assert "job" in out
    assert "shelved" in out
