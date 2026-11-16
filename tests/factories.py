"""
Deterministic builders shared across the HireSynth test suite.

Nothing here touches the network: skill embeddings come from a tiny one-hot
FakeModel instead of SentenceTransformer, and FrozenDateTime pins the clock
for signals that depend on "now".
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from utils.schema import (
    CandidateFlags,
    CandidateNormalized,
    ExperienceEntry,
    JDIntent,
    ScoredCandidate,
)

FROZEN_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)

# Every fixture skill gets its own axis, so cosine similarity is exactly 1.0
# for the same skill and 0.0 for any different skill.
SKILL_VOCAB: dict[str, int] = {
    "python": 0,
    "pytorch": 1,
    "docker": 2,
    "kubernetes": 3,
    "sql": 4,
    "excel": 5,
    "photoshop": 6,
}

DEFAULT_TEST_WEIGHTS: dict[str, float] = {
    "skills": 0.40,
    "trajectory": 0.20,
    "recency": 0.15,
    "behavioral": 0.15,
    "seniority": 0.10,
}

ALL_FIELDS = ["skills", "experience", "education", "location"]


class FakeModel:
    """Stand-in for SentenceTransformer that returns one-hot skill vectors."""

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        """Encode each skill string as a one-hot vector over SKILL_VOCAB."""
        out = np.zeros((len(texts), len(SKILL_VOCAB)), dtype="float32")
        for i, text in enumerate(texts):
            out[i, SKILL_VOCAB[text.lower()]] = 1.0  # KeyError means a fixture bug
        return out


class FrozenDateTime(datetime):
    """datetime subclass whose now() always returns FROZEN_NOW."""

    @classmethod
    def now(cls, tz=None):
        """Return the frozen instant (tz-aware UTC unless another tz is given)."""
        return FROZEN_NOW if tz is None else FROZEN_NOW.astimezone(tz)


def make_jd(**overrides) -> JDIntent:
    """Build a Senior ML Engineer JDIntent with explicit default weights."""
    fields = dict(
        job_id="JD_TEST",
        job_title="Senior ML Engineer",
        must_have=["Python", "PyTorch", "Docker"],
        nice_have=["Kubernetes"],
        seniority="senior",
        domain="machine_learning",
        culture_signals=[],
        implicit_needs=[],
        retrieval_mode="technical",
        weights=dict(DEFAULT_TEST_WEIGHTS),
        quality_score=85.0,
        quality_level="HIGH",
    )
    fields.update(overrides)
    return JDIntent(**fields)


def make_candidate(candidate_id: str, **overrides) -> CandidateNormalized:
    """Build a CandidateNormalized with a complete profile unless overridden."""
    fields = dict(
        candidate_id=candidate_id,
        raw_text="",
        skills=[],
        experience=[],
        total_years_exp=None,
        trajectory_label=None,
        fields_present=list(ALL_FIELDS),
        fields_missing=[],
        confidence_level="HIGH",
    )
    fields.update(overrides)
    return CandidateNormalized(**fields)


def current_role(title: str, start: str) -> ExperienceEntry:
    """An ongoing role that started at `start` (YYYY-MM)."""
    return ExperienceEntry(title=title, company="Acme", start_date=start, is_current=True)


def past_role(title: str, start: str, end: str) -> ExperienceEntry:
    """A finished role spanning `start`..`end` (YYYY-MM)."""
    return ExperienceEntry(title=title, company="Acme", start_date=start, end_date=end)


def make_scored(candidate_id: str, score: float) -> ScoredCandidate:
    """Build a minimal ScoredCandidate as Layer 2 would emit it."""
    return ScoredCandidate(
        rank=0,
        candidate_id=candidate_id,
        final_score=score,
        signal_fusion_score=score,
        llm_rerank_score=None,
        signal_breakdown={},
        confidence_level="HIGH",
        flags=CandidateFlags(llm_rerank=False),
        tier="strong_match",
        reasoning=None,
    )
