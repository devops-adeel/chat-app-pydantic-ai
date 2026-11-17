"""Graphiti temporal-knowledge-graph memory layer.

Storage is local and self-hosted: Neo4j running in a local Docker container (data
stays on the user's machine). Intelligence is rented: Claude does entity/edge
extraction, OpenAI does embeddings (text-embedding-3-small) and reranking (the
gpt-4o-mini LLM reranker).

(We first tried Kùzu — embedded, no container — but it is deprecated/unmaintained in
graphiti-core 0.29.2 and crashes on group_id ingestion. Neo4j is Graphiti's default
and best-supported driver; group_id is a property filter, so per-user scoping works
on Community Edition.)

Why a temporal graph (not vector recall): the user needs to reason over *evolving,
self-contradicting* personal facts (job switches, shifting priorities, energy levels
— the user has AuDHD). Graphiti tracks fact validity on edges and invalidates
superseded facts, so "what's true now" never returns a stale answer.

The transient/durable split (the "hybrid capture" decision) is implemented with:
  - custom EDGE types: transient `Feels`/`EnergyLevel` vs durable
    `Prefers`/`HoldsOpinion`/`WorksOn`/`HasHabit`,
  - `custom_extraction_instructions` that tell the extractor to treat mood/energy as
    volatile, and
  - retrieval-time filtering (`search_memory(..., durable_only=True)`) so a bad-day
    "I hate my thesis" never surfaces as a stable belief.

See the plan at ~/.claude/plans/let-s-walk-through-the-sleepy-crescent.md.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_filters import SearchFilters
from pydantic import BaseModel, Field

# Quiet Neo4j's "property key does not exist" notifications. Graphiti's queries
# reference properties (entity_edges, fact_embedding, …) that the planner can't
# statically verify on a sparse graph, so the driver logs a WARNING per query.
# They are benign (not query failures); silencing keeps real output readable.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Ontology — custom EDGE (fact) types.
#
# Graphiti stores temporal validity (valid_at / invalid_at) on EDGES, set by the
# LLM from temporal cues in the text — NOT on node types. So the transient/durable
# behaviour lives here, on the relations, plus the extraction instructions below
# and the retrieval filter in search_memory().
# ---------------------------------------------------------------------------


# --- Transient: volatile state that must NEVER surface as a stable belief ---
class Feels(BaseModel):
    """A transient emotional / mood state the user expressed (volatile)."""

    sentiment: str | None = Field(
        None, description="e.g. positive, negative, anxious, energised, low"
    )


class EnergyLevel(BaseModel):
    """A transient self-reported energy / capacity level (volatile; AuDHD-relevant)."""

    level: str | None = Field(None, description="e.g. high, low, depleted, wired, flat")


# --- Durable: safe to surface as 'currently true about the user' ------------
class Prefers(BaseModel):
    """A durable preference the user holds."""


class HoldsOpinion(BaseModel):
    """A durable opinion, belief, or stance the user holds."""


class WorksOn(BaseModel):
    """A durable project, focus, or commitment the user is working on (e.g. thesis)."""


class HasHabit(BaseModel):
    """A durable habit or routine the user has."""


# --- Structural entity types ------------------------------------------------
class Project(BaseModel):
    """A project, piece of work, or the user's thesis."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Project": Project,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {
    "Feels": Feels,
    "EnergyLevel": EnergyLevel,
    "Prefers": Prefers,
    "HoldsOpinion": HoldsOpinion,
    "WorksOn": WorksOn,
    "HasHabit": HasHabit,
}

TRANSIENT_EDGE_TYPES: list[str] = ["Feels", "EnergyLevel"]
DURABLE_EDGE_TYPES: list[str] = ["Prefers", "HoldsOpinion", "WorksOn", "HasHabit"]

# Which edge types may connect which (source, target) entity-type pairs.
# ("Entity", "Entity") is Graphiti's catch-all — kept permissive for the skeleton.
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Entity", "Entity"): list(EDGE_TYPES.keys()),
}

CUSTOM_EXTRACTION_INSTRUCTIONS = (
    "These episodes are a personal-assistant conversation with a single user who has "
    "AuDHD. Classify momentary emotional or energy statements (e.g. 'I feel useless "
    "today', 'no energy left', 'feeling great about X right now') as TRANSIENT "
    "relations (Feels, EnergyLevel): set valid_at to the episode time and prefer a "
    "near-term invalid_at, because these are volatile and must not be treated as "
    "lasting beliefs. Only use a DURABLE relation (Prefers, HoldsOpinion, WorksOn, "
    "HasHabit) when the user expresses a lasting stance, ongoing project, or settled "
    "preference — not a passing mood."
)


# ---------------------------------------------------------------------------
# Settings + client factory
# ---------------------------------------------------------------------------
@dataclass
class GraphitiSettings:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_api_key: str
    anthropic_api_key: str
    extraction_model: str
    embedding_model: str
    embedding_dim: int
    rerank_model: str


def settings_from_env() -> GraphitiSettings:
    return GraphitiSettings(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        # OpenAI powers embeddings + reranking (one key for both).
        openai_api_key=os.environ["OPENAI_API_KEY"],
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        extraction_model=os.environ.get("MEMORY_EXTRACTION_MODEL", "claude-sonnet-4-6"),
        embedding_model=os.environ.get(
            "MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        # text-embedding-3-small returns 1536 dims. Graphiti TRUNCATES embeddings to
        # this value by raw slicing (not OpenAI's `dimensions` re-normalization), so a
        # smaller value silently degrades retrieval quality — keep the full 1536.
        embedding_dim=int(os.environ.get("MEMORY_EMBEDDING_DIM", "1536")),
        # gpt-4o-mini is a chat model — it is the LLM *reranker*, not the embedder.
        rerank_model=os.environ.get("MEMORY_RERANK_MODEL", "gpt-4o-mini"),
    )


def build_graphiti(settings: GraphitiSettings | None = None) -> Graphiti:
    """Construct a Graphiti client: local Neo4j storage, rented Claude + OpenAI.

    Cheap to call; does NOT touch the database. Call `init_graphiti()` once at
    startup to create indices/constraints, and `close_graphiti()` on shutdown.
    """
    settings = settings or settings_from_env()
    llm_client = AnthropicClient(
        config=LLMConfig(
            api_key=settings.anthropic_api_key, model=settings.extraction_model
        )
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=settings.openai_api_key,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        )
    )
    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(api_key=settings.openai_api_key, model=settings.rerank_model)
    )
    # Neo4j is Graphiti's default driver when uri/user/password are passed.
    return Graphiti(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


async def init_graphiti(graphiti: Graphiti) -> None:
    """Create indices and constraints. Idempotent; safe to call on every startup."""
    await graphiti.build_indices_and_constraints()


async def close_graphiti(graphiti: Graphiti) -> None:
    """Close the Neo4j connection pool. Call on app shutdown."""
    await graphiti.close()


# ---------------------------------------------------------------------------
# Admission gate
#
# Ingesting EVERY turn is the most-cited cost/quality anti-pattern in agent memory:
# each episode costs ~5-9 extraction LLM calls + embeddings, and trivial turns
# ("thanks", "ok") just pollute the graph with garbage edges. This gate is
# deliberately *conservative* — it skips only obvious throwaway turns and errs
# toward ingesting, because terse-but-important facts ("I quit my job") must pass.
# A cheap LLM (Haiku) triage is the principled next step; this heuristic is free.
# ---------------------------------------------------------------------------
_TRIVIAL_TURNS = frozenset(
    {
        "thanks",
        "thank you",
        "thx",
        "ty",
        "ok",
        "okay",
        "k",
        "kk",
        "cool",
        "nice",
        "great",
        "got it",
        "sure",
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "lol",
        "haha",
        "hi",
        "hello",
        "hey",
        "bye",
        "goodbye",
        "good night",
        "morning",
    }
)


def should_ingest(user_msg: str) -> bool:
    """Return True if this turn is worth committing to long-term memory.

    Conservative: only filters out exact pleasantries / ultra-short acks. Anything
    with substance passes (favour recall of facts over saving LLM cost).
    """
    normalised = user_msg.strip().lower().rstrip("!.?")
    if len(normalised) < 3:
        return False
    return normalised not in _TRIVIAL_TURNS


# ---------------------------------------------------------------------------
# Ingestion + retrieval helpers
# ---------------------------------------------------------------------------
async def ingest_turn(
    graphiti: Graphiti,
    *,
    user_id: str,
    episode_uuid: str,
    episode_body: str,
    reference_time: datetime,
    previous_episode_uuid: str | None = None,
) -> tuple[str, list[str]]:
    """Ingest one chat turn as a Graphiti episode (sequential, contradiction-aware).

    MUST be called serially per user (the caller enforces this with a Restate
    Virtual Object keyed by user_id) — `add_episode` performs edge invalidation that
    depends on prior graph state.

    Idempotent per `episode_uuid`: a retry after a crash mid-ingest reuses the same
    episode node instead of minting a duplicate (Graphiti never dedupes episodes).
    Entities/facts are re-extracted on retry and converge via Graphiti's own dedup —
    best-effort, not exactly-once — and a fact invalidated by the interrupted attempt
    is not re-reported, so that reflection can be lost.

    Returns (episode_uuid, reflection_notes) where reflection_notes describe facts
    this turn invalidated — the proactive-nudge surface.
    """
    await _ensure_episode(
        graphiti,
        episode_uuid=episode_uuid,
        user_id=user_id,
        episode_body=episode_body,
        reference_time=reference_time,
    )
    results = await graphiti.add_episode(
        name=f"chat-{reference_time.isoformat()}",
        episode_body=episode_body,
        source=EpisodeType.message,
        source_description="chat turn",
        reference_time=reference_time,
        group_id=user_id,
        uuid=episode_uuid,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
        custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
        # [] not None: None makes Graphiti look up recent episodes by time, which
        # would return the episode we just pre-saved as its own "previous" context.
        previous_episode_uuids=(
            [previous_episode_uuid] if previous_episode_uuid else []
        ),
    )
    return results.episode.uuid, _reflections_from_edges(results.edges)


async def _ensure_episode(
    graphiti: Graphiti,
    *,
    episode_uuid: str,
    user_id: str,
    episode_body: str,
    reference_time: datetime,
) -> None:
    """Create the episode node once; never overwrite it.

    In graphiti-core 0.29.2, `add_episode(uuid=...)` *loads* an existing episode and
    fails if it is absent, so the node must exist first. Create-only (not upsert),
    matching Graphiti's own concurrent-ingestion design: overwriting on retry would
    reset the `entity_edges` an earlier attempt already linked.
    """
    try:
        await EpisodicNode.get_by_uuid(graphiti.driver, episode_uuid)
        return
    except NodeNotFoundError:
        pass
    episode = EpisodicNode(
        uuid=episode_uuid,
        name=f"chat-{reference_time.isoformat()}",
        group_id=user_id,
        labels=[],
        source=EpisodeType.message,
        source_description="chat turn",
        content=episode_body,
        created_at=datetime.now(tz=UTC),
        valid_at=reference_time,
    )
    await episode.save(graphiti.driver)


def _reflections_from_edges(edges: list[EntityEdge]) -> list[str]:
    """A change worth reflecting back = a fact that just gained an end-of-validity.

    `add_episode` does not return a separate `invalidated_edges` list, but every
    superseded fact carries `invalid_at`/`expired_at`, so we surface those.
    """
    notes: list[str] = []
    for edge in edges:
        if edge.invalid_at or edge.expired_at:
            notes.append(f"changed: {edge.fact}")
    return notes


async def search_memory(
    graphiti: Graphiti,
    *,
    user_id: str,
    query: str,
    durable_only: bool = True,
    num_results: int = 8,
) -> list[str]:
    """Return relevant fact strings for the user, scoped to their graph.

    durable_only=True filters OUT transient Feels/EnergyLevel edges, so volatile
    mood/energy is never surfaced as a stable belief.
    """
    search_filter = (
        SearchFilters(edge_types=DURABLE_EDGE_TYPES) if durable_only else None
    )
    edges = await graphiti.search(
        query=query,
        group_ids=[user_id],
        num_results=num_results,
        search_filter=search_filter,
    )
    return [edge.fact for edge in edges]
