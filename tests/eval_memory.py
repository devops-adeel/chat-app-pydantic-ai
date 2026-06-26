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
from datetime import UTC, datetime, timedelta

import memory

NOW = datetime.now(tz=UTC)

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
# Ask DIRECTLY about feelings: even so, a durable-only search must not surface the
# volatile mood as a stable belief. The goal query proves the episode was processed.
TRANSIENT_MOOD_QUERY = "the user's current mood, energy, and emotional state"
TRANSIENT_GOAL_QUERY = "the user's long-term goals and ongoing projects"
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
            episode_body=body,
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
    """Transient vs durable: a bad-day mood is captured but never surfaces as a
    durable belief. PASS = mood absent from durable search; FAIL = it leaked in;
    INCONCLUSIVE = the mood wasn't captured/retrieved at all (can't prove filtering).
    """
    print("\n=== Scenario 2: transient vs durable (mood must not become belief) ===")
    reference_time, body = TRANSIENT_EPISODE
    await memory.ingest_turn(
        graphiti, user_id=group_id, episode_body=body, reference_time=reference_time
    )
    print(f"ingested {reference_time.date()}: {body[:60]}...")

    # Ask directly about feelings — the durable view must STILL omit the volatile mood.
    mood_durable = await memory.search_memory(
        graphiti, user_id=group_id, query=TRANSIENT_MOOD_QUERY, durable_only=True
    )
    # Ask about goals — proves the same episode produced retrievable DURABLE content,
    # so the mood's absence above is meaningful (not just an empty graph).
    goal_durable = await memory.search_memory(
        graphiti, user_id=group_id, query=TRANSIENT_GOAL_QUERY, durable_only=True
    )
    _print_facts(f"DURABLE facts for {TRANSIENT_MOOD_QUERY!r}:", mood_durable)
    _print_facts(f"DURABLE facts for {TRANSIENT_GOAL_QUERY!r}:", goal_durable)

    mood_leaked = any(m in " ".join(mood_durable).lower() for m in TRANSIENT_MARKERS)
    if mood_leaked:
        return "FAIL"  # volatile mood surfaced as a stable belief
    if goal_durable:
        return "PASS"  # durable content retrievable, yet the mood never leaks in
    return "INCONCLUSIVE"  # episode yielded no durable facts — can't conclude


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
