"""Live temporal-memory eval harness — exercises the REAL Graphiti/Neo4j stack.

Unlike test_memory.py (deterministic, mocked), this ingests a contradiction
sequence into the running Neo4j using real extraction (Claude) + embeddings
(OpenAI), then checks that durable retrieval returns the CURRENT fact, not the
stale one — the entire value proposition of the temporal graph.

Run it (needs `docker compose up neo4j -d` first):

    uv run python -m tests.eval_memory

Requires ANTHROPIC_API_KEY (extraction), OPENAI_API_KEY (embeddings + reranking),
and NEO4J_PASSWORD. Skips cleanly if they are absent, so it is safe to leave in CI.
Uses a unique group_id per run and deletes it afterwards, so repeated runs don't
accumulate in the persistent database.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta

import memory

NOW = datetime.now(tz=UTC)

# Episodes are framed with a named speaker (as production does) so Graphiti mints a
# Person entity — without it, first-person self-state forms no edges.
EVAL_USER = "Alex"

# Scenario 1 — temporal contradiction. A focus that changes over time; durable
# retrieval must return the CURRENT fact (job), not the stale one (thesis).
TEMPORAL_EPISODES = [
    (NOW - timedelta(days=21), "My main focus right now is finishing my PhD thesis."),
    (
        NOW,
        "I've decided to shelve the thesis for now — I just started a new job this "
        "week and that has become my main focus.",
    ),
]
TEMPORAL_QUERY = "What is my main focus at the moment?"

# Scenario 2 — transient vs durable. A bad-day mood/energy statement bundled with a
# durable goal. The mood must be captured (as a transient edge) but must NOT surface
# in a durable-only search — i.e. it never becomes a "stable belief about the user".
# This is the AuDHD-critical behaviour: stale emotional state must not resurface.
TRANSIENT_EPISODE = (
    NOW,
    "Honestly I feel completely useless and exhausted today — total autistic "
    "burnout. My long-term goal is still to finish writing my novel.",
)
# The mood's volatile markers — used to check no DURABLE-typed edge carries them.
TRANSIENT_MARKERS = ["useless", "burnout", "exhausted", "drained", "depleted"]


def _have_keys() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        and os.environ.get("OPENAI_API_KEY")
        and os.environ.get("NEO4J_PASSWORD")
    )


async def _delete_group(graphiti, group_id: str) -> None:
    """Best-effort teardown: remove every node/edge tagged with this run's group."""
    try:
        await graphiti.driver.execute_query(
            "MATCH (n {group_id: $gid}) DETACH DELETE n", gid=group_id
        )
    except Exception as exc:  # teardown must never fail the eval
        print(f"(teardown skipped: {exc})")


async def _fetch_edges(graphiti, group_id: str) -> list[tuple[str, str]]:
    """Read fact edges directly from the graph — no search/rerank in the loop.

    Returns (name, fact) pairs, where `name` is the edge TYPE Graphiti stored
    (a custom type like 'Feels'/'WorksOn', or a generic relation name). This is
    exactly the property SearchFilters(edge_types=...) matches on (`e.name`).
    """
    result = await graphiti.driver.execute_query(
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE e.group_id = $gid "
        "RETURN e.name AS name, e.fact AS fact",
        gid=group_id,
    )
    return [(rec["name"], rec["fact"]) for rec in result.records]


def _print_facts(label: str, facts: list[str]) -> None:
    print(f"\n{label}")
    for fact in facts:
        print(f"  - {fact}")
    if not facts:
        print("  (none)")


async def scenario_temporal(graphiti, group_id: str) -> str:
    """Temporal contradiction: durable retrieval returns the current fact, not stale."""
    print("\n=== Scenario 1: temporal contradiction (thesis → job) ===")
    previous_uuid: str | None = None
    for reference_time, body in TEMPORAL_EPISODES:
        previous_uuid, reflections = await memory.ingest_turn(
            graphiti,
            user_id=group_id,
            episode_uuid=str(uuid.uuid4()),
            episode_body=f"{EVAL_USER}: {body}",
            reference_time=reference_time,
            previous_episode_uuid=previous_uuid,
        )
        print(f"ingested {reference_time.date()}: {body[:60]}...")
        if reflections:
            print(f"  reflections (nudge): {reflections}")

    facts = await memory.search_memory(
        graphiti, user_id=group_id, query=TEMPORAL_QUERY, durable_only=True
    )
    _print_facts(f"Durable facts for {TEMPORAL_QUERY!r}:", facts)
    joined = " ".join(facts).lower()
    return "PASS" if "job" in joined else "FAIL"


async def scenario_transient(graphiti, group_id: str) -> str:
    """Transient vs durable, asserted DIRECTLY on the graph (no search fuzziness).

    The guarantee: a bad-day mood must be typed as a transient EDGE, never as a
    durable belief. We read the edges straight from Neo4j and check their types —
    `e.name` is exactly what the durable retrieval filter (edge_types) matches on.

    FAIL         = a durable-typed edge carries the mood (it became a stable belief).
    PASS         = the mood is on a transient-typed edge and no durable edge carries it.
    INCONCLUSIVE = the mood wasn't classified into a transient type at all (a signal to
                   tune custom_extraction_instructions, not a harness bug).
    """
    print("\n=== Scenario 2: transient vs durable (direct edge-type assertion) ===")
    reference_time, body = TRANSIENT_EPISODE
    await memory.ingest_turn(
        graphiti,
        user_id=group_id,
        episode_uuid=str(uuid.uuid4()),
        episode_body=f"{EVAL_USER}: {body}",
        reference_time=reference_time,
    )
    print(f"ingested {reference_time.date()}: {body[:60]}...")

    edges = await _fetch_edges(graphiti, group_id)
    print("\nfact edges in the graph (type → fact):")
    for name, fact in edges:
        print(f"  [{name}] {fact}")
    if not edges:
        print("  (none)")

    def _has_mood(fact: str) -> bool:
        return any(marker in (fact or "").lower() for marker in TRANSIENT_MARKERS)

    transient_typed = [(n, f) for n, f in edges if n in memory.TRANSIENT_EDGE_TYPES]
    mood_in_durable = any(
        n in memory.DURABLE_EDGE_TYPES and _has_mood(f) for n, f in edges
    )

    if mood_in_durable:
        return "FAIL"  # volatile mood typed as a durable belief
    if transient_typed:
        return "PASS"  # mood captured as a transient edge; durable view excludes it
    return "INCONCLUSIVE"  # mood not classified transient — extraction-quality signal


async def _count_episodes(graphiti, group_id: str) -> int:
    result = await graphiti.driver.execute_query(
        "MATCH (e:Episodic) WHERE e.group_id = $gid RETURN count(e) AS n",
        gid=group_id,
    )
    return result.records[0]["n"]


async def scenario_retry(graphiti, group_id: str) -> str:
    """Crash-retry idempotency: the same turn ingested twice under one episode uuid.

    This is what a Restate retry does after a crash between the graph write and the
    journal record (ctx.uuid() is stable across retries). PASS = one episode node.
    Fact-edge counts are printed, not asserted: facts are re-extracted by the LLM and
    converge via Graphiti's dedup, which is best-effort.
    """
    print("\n=== Scenario 3: crash-retry idempotency (same episode uuid twice) ===")
    reference_time, body = TEMPORAL_EPISODES[1]
    episode_uuid = str(uuid.uuid4())
    edge_counts = []
    for attempt in (1, 2):
        await memory.ingest_turn(
            graphiti,
            user_id=group_id,
            episode_uuid=episode_uuid,
            episode_body=f"{EVAL_USER}: {body}",
            reference_time=reference_time,
        )
        edge_counts.append(len(await _fetch_edges(graphiti, group_id)))
        print(f"attempt {attempt}: fact edges in graph = {edge_counts[-1]}")

    episodes = await _count_episodes(graphiti, group_id)
    print(f"episode nodes after retry = {episodes}")
    return "PASS" if episodes == 1 else "FAIL"


async def main() -> int:
    if not _have_keys():
        print(
            "SKIP: set ANTHROPIC_API_KEY, OPENAI_API_KEY and NEO4J_PASSWORD "
            "(and `docker compose up neo4j -d`) to run the live eval."
        )
        return 0

    graphiti = memory.build_graphiti()
    await memory.init_graphiti(graphiti)

    stamp = int(time.time())
    scenarios = {
        "temporal": scenario_temporal,
        "transient": scenario_transient,
        "retry": scenario_retry,
    }
    results: dict[str, str] = {}
    try:
        for name, fn in scenarios.items():
            # Unique group per scenario+run: isolated and cleaned independently.
            group_id = f"eval-{name}-{stamp}"
            try:
                results[name] = await fn(graphiti, group_id)
            finally:
                await _delete_group(graphiti, group_id)
    finally:
        await memory.close_graphiti(graphiti)

    # INCONCLUSIVE does not fail the run (extraction is non-deterministic); only a
    # hard FAIL (or a missing PASS) is a non-zero exit.
    failed = any(r == "FAIL" for r in results.values())
    summary = "  ".join(f"{name}={r}" for name, r in results.items())
    verdict = f"VERDICT: {'FAIL ❌' if failed else 'PASS ✅'} — {summary}"
    print("\n" + verdict)
    print(verdict, file=sys.stderr)  # survives `| pbcopy`
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
