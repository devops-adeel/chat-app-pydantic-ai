"""Crash-inside-the-step test against a REAL Restate server (Docker).

The journal records a step's result only once the step completes. If the process
dies after the graph write but before that record, Restate re-runs the step. This
test injects exactly that failure into `MemoryService.ingest` and checks the retry
reuses the same episode instead of minting a duplicate.

Graphiti is faked (no API keys, no Neo4j); Restate is real. Skips without Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

os.environ.setdefault("TAVILY_API_KEY", "dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

pytest.importorskip("testcontainers")
pytest.importorskip("hypercorn")

import restate  # noqa: E402
from restate.harness import create_test_harness  # noqa: E402
from test_memory import FakeEpisodeStore, fake_graphiti  # noqa: E402

import restate_service as rs  # noqa: E402

# The image docker-compose.yml pins, so the test exercises the runtime we ship.
RESTATE_IMAGE = "docker.io/restatedev/restate:1.7.0"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _docker_available(), reason="needs Docker")


async def test_crash_after_write_is_retried_onto_the_same_episode(monkeypatch):
    store = FakeEpisodeStore(fail_first_add_after_write=True)
    store.install(monkeypatch)

    async def get_graphiti():
        return fake_graphiti(store)

    monkeypatch.setattr(rs, "get_graphiti", get_graphiti)
    app = restate.app(services=[rs.memory_service])

    async with create_test_harness(app, restate_image=RESTATE_IMAGE) as env:
        result = await env.client.object_call(
            rs.ingest,
            key="test-user",
            arg={
                "episode_body": "Alex: I started a new job.",
                "reference_time": "2026-11-17T00:00:00+00:00",
            },
        )

    attempts = [call["uuid"] for call in store.add_calls]
    assert len(attempts) == 2, "the injected crash should force exactly one retry"
    assert attempts[0] == attempts[1] == result["episode_uuid"]
    assert list(store.episodes) == [result["episode_uuid"]]  # one episode, not two
    assert store.saves == 1  # the retry found the node and did not overwrite it
