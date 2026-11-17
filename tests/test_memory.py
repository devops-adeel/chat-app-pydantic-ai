"""Wiring + temporal-behaviour tests for the Graphiti memory layer.

These are deterministic and need no API keys or database: the model is
TestModel/FunctionModel and `memory.search_memory` is monkeypatched. They verify
the *wiring* and the transient/durable contract — not Graphiti's extraction quality
(that needs the live harness in `eval_memory.py` against a real graph).
"""

from __future__ import annotations

import os
import types
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import Any

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


# --- Idempotent ingestion under Restate retries ----------------------------
class FakeEpisodeStore:
    """Mimics graphiti-core 0.29.2: `add_episode(uuid=X)` loads episode X and raises
    NodeNotFoundError when it is absent; Graphiti never dedupes episode nodes."""

    def __init__(self, fail_first_add_after_write: bool = False):
        self.episodes: dict[str, Any] = {}
        self.saves = 0
        self.add_calls: list[dict] = []
        self.fail_first_add_after_write = fail_first_add_after_write

    def install(self, monkeypatch) -> None:
        store = self

        async def get_by_uuid(cls, driver, uuid):
            if uuid not in store.episodes:
                raise memory.NodeNotFoundError(uuid)
            return store.episodes[uuid]

        async def save(node, driver):
            store.saves += 1
            store.episodes[node.uuid] = node

        monkeypatch.setattr(
            memory.EpisodicNode, "get_by_uuid", classmethod(get_by_uuid)
        )
        monkeypatch.setattr(memory.EpisodicNode, "save", save)

    async def add_episode(self, **kwargs):
        self.add_calls.append(kwargs)
        episode = self.episodes[kwargs["uuid"]]  # KeyError ≈ NodeNotFoundError
        episode.entity_edges = ["edge-1"]  # the write an interrupted attempt leaves
        if self.fail_first_add_after_write and len(self.add_calls) == 1:
            raise ConnectionError("crash after the graph write, before journaling")
        return types.SimpleNamespace(episode=episode, edges=[])


def fake_graphiti(store: FakeEpisodeStore):
    return types.SimpleNamespace(driver=object(), add_episode=store.add_episode)


async def _ingest(graphiti, previous: str | None = None):
    return await memory.ingest_turn(
        graphiti,
        user_id="test-user",
        episode_uuid="ep-1",
        episode_body="Alex: I started a new job.\nAssistant: Congrats!",
        reference_time=datetime(2026, 11, 17, tzinfo=UTC),
        previous_episode_uuid=previous,
    )


async def test_ingest_turn_retry_reuses_one_episode(monkeypatch):
    store = FakeEpisodeStore()
    store.install(monkeypatch)
    graphiti = fake_graphiti(store)

    await _ingest(graphiti)
    await _ingest(graphiti)  # the retry

    assert list(store.episodes) == ["ep-1"]
    assert store.saves == 1  # create-only: the retry never overwrites
    assert store.episodes["ep-1"].entity_edges == ["edge-1"]
    assert all(call["uuid"] == "ep-1" for call in store.add_calls)


async def test_first_ingest_never_lets_graphiti_look_up_previous_episodes(
    monkeypatch,
):
    # None would make Graphiti fetch recent episodes by time — including the one
    # just pre-saved — as this episode's own context.
    store = FakeEpisodeStore()
    store.install(monkeypatch)

    await _ingest(fake_graphiti(store), previous=None)
    await _ingest(fake_graphiti(store), previous="ep-0")

    assert store.add_calls[0]["previous_episode_uuids"] == []
    assert store.add_calls[1]["previous_episode_uuids"] == ["ep-0"]


class FakeContext:
    """Just enough of a Restate context to drive the handlers in-process."""

    def __init__(self, key: str = "test-user", now: float = 1_763_337_600.0):
        self._key, self._now = key, now
        self.state: dict = {}
        self.sent: list[dict] = []

    def key(self) -> str:
        return self._key

    def uuid(self):
        return uuid_mod.UUID(int=42)  # deterministic, as Restate's is per invocation

    async def time(self) -> float:
        return self._now

    async def get(self, name, type_hint=None):
        return self.state.get(name)

    def set(self, name, value) -> None:
        self.state[name] = value

    async def run_typed(self, name, fn):
        return await fn()

    async def object_call(self, handler, key, arg):
        return []

    def object_send(self, handler, key, arg) -> None:
        self.sent.append(arg)


async def test_ingest_handler_passes_a_stable_episode_uuid(monkeypatch):
    store = FakeEpisodeStore()
    store.install(monkeypatch)

    async def get_graphiti():
        return fake_graphiti(store)

    monkeypatch.setattr(rs, "get_graphiti", get_graphiti)
    payload = {
        "episode_body": "Alex: I started a new job.",
        "reference_time": "2026-11-17T00:00:00+00:00",
    }
    ctx: Any = FakeContext()
    result = await rs.ingest(ctx, payload)

    assert result["episode_uuid"] == str(uuid_mod.UUID(int=42))
    assert store.add_calls[0]["uuid"] == result["episode_uuid"]


async def test_chat_run_uses_the_durable_clock_for_ingestion(monkeypatch):
    async def get_graphiti():
        return object()

    async def agent_run(prompt, **kwargs):
        return types.SimpleNamespace(output="Congrats!", new_messages=lambda: [])

    monkeypatch.setattr(rs, "get_graphiti", get_graphiti)
    monkeypatch.setattr(rs.restate_agent, "run", agent_run)
    ctx: Any = FakeContext(now=1_763_337_600.0)

    out = await rs.run(ctx, {"prompt": "I started a new job this week."})

    expected = datetime.fromtimestamp(1_763_337_600.0, tz=UTC).isoformat()
    assert ctx.sent[0]["reference_time"] == expected  # replay-safe send argument
    assert out["timestamp"] == expected
