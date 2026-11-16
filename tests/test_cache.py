"""
Tests for the shared LLM cache (data/llm_cache.json).

Two writers share one file:
  — Layer 0 JD-parse cache   → "jd_parse"      section, keyed by md5(jd_text)[:8]
  — Layer 3 re-rank cache    → "llm_responses" section, keyed by
                               md5(jd_text + sorted(candidate_ids))[:8]

Every test uses the isolated_llm_cache fixture, so the real cache file in
data/ is never read or written.  OpenAI is never called: tests that exercise
the miss path replace the client and the async re-rank call with fakes.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

import pipeline.layer0_normalizer as layer0
import pipeline.layer3_reranker as layer3
from tests.factories import make_jd, make_scored
from utils.schema import ReasoningCard

JD_TEXT = (
    "Senior ML Engineer. Must have Python, PyTorch and Docker; Kubernetes is a "
    "plus. 5+ years building production machine learning systems at a startup."
)

CANDIDATE_IDS = ["C003", "C001", "C002"]

CACHED_RESPONSES = {
    "C001": {"strength": "Strong PyTorch.", "gap": "No Kubernetes.",
             "recommendation": "Interview.", "llm_score": 9},
    "C002": {"strength": "Solid Python.", "gap": "Junior.",
             "recommendation": "Phone screen.", "llm_score": 6},
    "C003": {"strength": "Docker expert.", "gap": "Little ML.",
             "recommendation": "Hold.", "llm_score": 4},
}


def _scored_pool():
    """Layer 2 output for the three cached candidates, already ranked by score."""
    return [make_scored("C001", 85.0), make_scored("C002", 62.0), make_scored("C003", 41.0)]


@pytest.fixture
def no_openai(monkeypatch):
    """Fail loudly if any code path tries to build an OpenAI client."""
    def _boom(*_a, **_kw):
        raise AssertionError("OpenAI client constructed — cache was bypassed")

    monkeypatch.setattr(layer3, "AsyncOpenAI", _boom)
    monkeypatch.setattr(layer0, "_call_llm", _boom)


@pytest.fixture
def fake_rerank(monkeypatch):
    """Replace the network-bound re-rank call; records each invocation."""
    calls: list[list[str]] = []

    async def _fake(candidates_with_summaries, jd_intent, semaphore, client):
        calls.append([sc.candidate_id for sc, _ in candidates_with_summaries])
        return [
            (sc, ReasoningCard("fresh s", "fresh g", "fresh r", 7), 70.0, True)
            for sc, _ in candidates_with_summaries
        ]

    monkeypatch.setattr(layer3, "AsyncOpenAI", lambda *a, **kw: object())
    monkeypatch.setattr(layer3, "rerank_all_async", _fake)
    return calls


# ---------------------------------------------------------------------------
# Layer 3 — cache key
# ---------------------------------------------------------------------------

class TestLLMCacheKey:
    """The key must identify (JD, candidate set) and nothing else."""

    def test_is_8_hex_chars(self):
        key = layer3._compute_cache_key(JD_TEXT, CANDIDATE_IDS)
        assert len(key) == 8
        int(key, 16)

    def test_stable_value(self):
        # Pinned so an accidental change to the key recipe (which would
        # silently invalidate every existing cache) fails loudly.
        assert layer3._compute_cache_key("jd", ["B", "A"]) == "3ff43bb7"  # md5("jdAB")

    def test_independent_of_candidate_order(self):
        assert layer3._compute_cache_key(JD_TEXT, ["C1", "C2", "C3"]) == (
            layer3._compute_cache_key(JD_TEXT, ["C3", "C1", "C2"])
        )

    def test_changes_with_jd_text(self):
        assert layer3._compute_cache_key(JD_TEXT, CANDIDATE_IDS) != (
            layer3._compute_cache_key(JD_TEXT + " Remote OK.", CANDIDATE_IDS)
        )

    def test_changes_with_candidate_set(self):
        assert layer3._compute_cache_key(JD_TEXT, ["C1", "C2"]) != (
            layer3._compute_cache_key(JD_TEXT, ["C1", "C2", "C3"])
        )


# ---------------------------------------------------------------------------
# Layer 3 — load / save
# ---------------------------------------------------------------------------

class TestLLMCacheStore:
    """Round-trip and every documented miss condition."""

    def test_missing_file_is_a_miss(self, isolated_llm_cache):
        assert not isolated_llm_cache.exists()
        assert layer3._load_llm_cache("deadbeef") is None

    def test_round_trip(self, isolated_llm_cache):
        layer3._save_llm_cache("deadbeef", CACHED_RESPONSES)
        assert layer3._load_llm_cache("deadbeef") == CACHED_RESPONSES

    def test_wrong_key_is_a_miss(self, isolated_llm_cache):
        layer3._save_llm_cache("deadbeef", CACHED_RESPONSES)
        assert layer3._load_llm_cache("cafebabe") is None

    def test_version_mismatch_is_a_miss(self, isolated_llm_cache):
        layer3._save_llm_cache("deadbeef", CACHED_RESPONSES)
        data = json.loads(isolated_llm_cache.read_text())
        data["cache_version"] = "0.9"
        isolated_llm_cache.write_text(json.dumps(data))
        assert layer3._load_llm_cache("deadbeef") is None

    def test_corrupt_file_is_a_miss_and_is_overwritten(self, isolated_llm_cache):
        isolated_llm_cache.write_text("{not json")
        assert layer3._load_llm_cache("deadbeef") is None
        layer3._save_llm_cache("deadbeef", CACHED_RESPONSES)
        assert layer3._load_llm_cache("deadbeef") == CACHED_RESPONSES

    def test_save_creates_parent_directory(self, tmp_path, monkeypatch):
        path = tmp_path / "nested" / "dir" / "llm_cache.json"
        monkeypatch.setattr(layer3, "_CACHE_PATH", path)
        layer3._save_llm_cache("deadbeef", {})
        assert path.exists()

    def test_newer_run_replaces_older_one(self, isolated_llm_cache):
        layer3._save_llm_cache("aaaaaaaa", {"C1": {"llm_score": 1}})
        layer3._save_llm_cache("bbbbbbbb", {"C2": {"llm_score": 2}})
        assert layer3._load_llm_cache("aaaaaaaa") is None
        assert layer3._load_llm_cache("bbbbbbbb") == {"C2": {"llm_score": 2}}


# ---------------------------------------------------------------------------
# Layer 0 — JD parse cache
# ---------------------------------------------------------------------------

class TestJDCache:
    """JDIntent round-trips through the jd_parse section."""

    def test_key_is_md5_prefix(self):
        assert layer0._jd_cache_key("jd") == "5a544a1c"  # md5("jd")
        assert layer0._jd_cache_key(JD_TEXT) != layer0._jd_cache_key(JD_TEXT + " ")

    def test_round_trip(self, isolated_llm_cache):
        intent = make_jd(is_multi_role=True, primary_track={"skills": ["Python"]})
        layer0._save_jd_cache("deadbeef", intent)
        assert layer0._load_jd_cache("deadbeef") == intent

    def test_wrong_hash_is_a_miss(self, isolated_llm_cache):
        layer0._save_jd_cache("deadbeef", make_jd())
        assert layer0._load_jd_cache("cafebabe") is None

    def test_schema_drift_is_a_miss_not_a_crash(self, isolated_llm_cache):
        """An intent saved by an older JDIntent shape must not raise TypeError."""
        layer0._save_jd_cache("deadbeef", make_jd())
        data = json.loads(isolated_llm_cache.read_text())
        data["jd_parse"]["intent"]["field_from_old_version"] = 1
        isolated_llm_cache.write_text(json.dumps(data))
        assert layer0._load_jd_cache("deadbeef") is None

    def test_parse_jd_cache_hit_skips_llm(self, isolated_llm_cache, no_openai):
        cached = make_jd(job_id="JD_OLD")
        layer0._save_jd_cache(layer0._jd_cache_key(JD_TEXT), cached)

        result = layer0.parse_jd(JD_TEXT, "JD_NEW")

        assert result.job_id == "JD_NEW"  # call-site metadata, never cached
        assert dataclasses.replace(result, job_id="JD_OLD") == cached

    def test_parse_jd_miss_calls_llm_once_then_hits(self, isolated_llm_cache, monkeypatch):
        calls = []

        def _fake_llm(system, user, max_tokens=1024):
            calls.append(user)
            return json.dumps({
                "job_title": "Senior ML Engineer",
                "must_have": ["Python", "PyTorch", "Docker"],
                "nice_have": ["Kubernetes"],
                "seniority": "senior",
                "domain": "machine_learning",
                "culture_signals": ["startup"],
                "implicit_needs": [],
                "retrieval_mode": "technical",
                "weights": {"skills": 0.4, "trajectory": 0.2, "recency": 0.15,
                            "behavioral": 0.15, "seniority": 0.1},
            })

        monkeypatch.setattr(layer0, "_call_llm", _fake_llm)
        first = layer0.parse_jd(JD_TEXT, "JD_1")
        second = layer0.parse_jd(JD_TEXT, "JD_1")

        assert len(calls) == 1
        assert first == second
        assert first.must_have == ["Python", "PyTorch", "Docker"]


# ---------------------------------------------------------------------------
# Shared file — the two sections must not clobber each other
# ---------------------------------------------------------------------------

class TestSharedFile:
    """Layer 0 and Layer 3 upsert their own section of the same JSON file."""

    def test_llm_save_preserves_jd_section(self, isolated_llm_cache):
        intent = make_jd()
        layer0._save_jd_cache("11111111", intent)
        layer3._save_llm_cache("22222222", CACHED_RESPONSES)
        assert layer0._load_jd_cache("11111111") == intent
        assert layer3._load_llm_cache("22222222") == CACHED_RESPONSES

    def test_jd_save_preserves_llm_section(self, isolated_llm_cache):
        layer3._save_llm_cache("22222222", CACHED_RESPONSES)
        layer0._save_jd_cache("11111111", make_jd())
        assert layer3._load_llm_cache("22222222") == CACHED_RESPONSES


# ---------------------------------------------------------------------------
# run_layer3 — cache integration
# ---------------------------------------------------------------------------

class TestRunLayer3Cache:
    """Cache hit/miss/bypass behaviour of the Layer 3 orchestrator."""

    def _seed(self):
        key = layer3._compute_cache_key(JD_TEXT, CANDIDATE_IDS)
        layer3._save_llm_cache(key, CACHED_RESPONSES)
        return key

    def test_hit_uses_cached_reasoning_without_api(self, isolated_llm_cache, no_openai):
        self._seed()
        result = layer3.run_layer3(
            make_jd(), _scored_pool(), {}, top_k=10, use_cache=True, jd_text=JD_TEXT
        )
        by_id = {sc.candidate_id: sc for sc in result}
        assert by_id["C001"].reasoning.strength == "Strong PyTorch."
        assert by_id["C001"].llm_rerank_score == 90.0
        assert by_id["C002"].llm_rerank_score == 60.0
        assert all(sc.flags.llm_rerank for sc in result)

    def test_hit_never_changes_ranking(self, isolated_llm_cache, no_openai):
        """LLM scores are informational; order follows the fusion score."""
        self._seed()
        result = layer3.run_layer3(
            make_jd(), _scored_pool(), {}, top_k=10, use_cache=True, jd_text=JD_TEXT
        )
        assert [(sc.rank, sc.candidate_id, sc.final_score) for sc in result] == [
            (1, "C001", 85.0), (2, "C002", 62.0), (3, "C003", 41.0)
        ]
        # C003: llm 40 vs fusion 41 → no conflict; C002: 60 vs 62 → no conflict.
        # C001: 90 vs 85 → no conflict (threshold is > 15 points).
        assert not any(sc.flags.score_conflict for sc in result)

    def test_miss_calls_api_and_populates_cache(self, isolated_llm_cache, fake_rerank):
        result = layer3.run_layer3(
            make_jd(), _scored_pool(), {}, top_k=10, use_cache=True, jd_text=JD_TEXT
        )
        assert fake_rerank == [["C001", "C002", "C003"]]
        assert all(sc.reasoning.strength == "fresh s" for sc in result)

        key = layer3._compute_cache_key(JD_TEXT, CANDIDATE_IDS)
        cached = layer3._load_llm_cache(key)
        assert set(cached) == {"C001", "C002", "C003"}
        assert cached["C001"]["llm_score"] == 7

    def test_second_run_is_served_from_cache(self, isolated_llm_cache, fake_rerank):
        for _ in range(2):
            layer3.run_layer3(
                make_jd(), _scored_pool(), {}, top_k=10, use_cache=True, jd_text=JD_TEXT
            )
        assert len(fake_rerank) == 1

    def test_different_candidate_set_is_a_miss(self, isolated_llm_cache, fake_rerank):
        self._seed()
        pool = _scored_pool() + [make_scored("C004", 30.0)]
        layer3.run_layer3(make_jd(), pool, {}, top_k=10, use_cache=True, jd_text=JD_TEXT)
        assert fake_rerank == [["C001", "C002", "C003", "C004"]]

    def test_use_cache_false_bypasses_read_and_write(self, isolated_llm_cache, fake_rerank):
        key = self._seed()
        before = isolated_llm_cache.read_text()

        result = layer3.run_layer3(
            make_jd(), _scored_pool(), {}, top_k=10, use_cache=False, jd_text=JD_TEXT
        )

        assert len(fake_rerank) == 1
        assert all(sc.reasoning.strength == "fresh s" for sc in result)
        assert isolated_llm_cache.read_text() == before
        assert layer3._load_llm_cache(key) == CACHED_RESPONSES

    def test_failed_calls_are_not_cached(self, isolated_llm_cache, monkeypatch):
        async def _partial(candidates_with_summaries, jd_intent, semaphore, client):
            out = []
            for sc, _ in candidates_with_summaries:
                ok = sc.candidate_id != "C002"
                out.append((sc, ReasoningCard("s", "g", "r", 5), 50.0, ok))
            return out

        monkeypatch.setattr(layer3, "AsyncOpenAI", lambda *a, **kw: object())
        monkeypatch.setattr(layer3, "rerank_all_async", _partial)

        layer3.run_layer3(make_jd(), _scored_pool(), {}, top_k=10, jd_text=JD_TEXT)

        cached = layer3._load_llm_cache(layer3._compute_cache_key(JD_TEXT, CANDIDATE_IDS))
        assert set(cached) == {"C001", "C003"}
