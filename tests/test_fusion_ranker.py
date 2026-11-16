"""
Tests for the Layer 2 signal-fusion ranker (pipeline/layer2_scoring.py).

A fixed JD + fixed candidate pool must always produce the same ranking and
the same scores.  Expected scores are hand-derived from the documented signal
formulas (see the per-candidate comments) with the clock frozen at
2026-06-01 and one-hot skill embeddings, so a change in any weight, decay
table or threshold shows up here as a failing golden test.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from pipeline.layer2_scoring import fuse_signals, precompute_skill_embeddings, run_layer2
from tests.factories import (
    DEFAULT_TEST_WEIGHTS,
    FakeModel,
    current_role,
    make_candidate,
    make_jd,
    past_role,
)

# Weights: skills .40, trajectory .20, recency .15, behavioral .15, seniority .10
# JD: must_have = Python, PyTorch, Docker (3 → threshold 0.75); nice = Kubernetes
# Behavioral raw = .4*completeness + .2*github + .2*openness + .2*exp_recency


def _pool():
    """Six candidates covering strong / gap / good / overqualified / poor / fresher."""
    return [
        # skills 1.0 | traj 1.0 | recency 1.0 | behav .4+.2+.2*.5(41mo)+.2 = .90
        # seniority 6y in [4,10] → 1.0
        # .40 + .20 + .15 + .135 + .10 = 98.50
        make_candidate(
            "C_STRONG",
            skills=["Python", "PyTorch", "Docker", "Kubernetes"],
            trajectory_label="high_growth",
            total_years_exp=6,
            experience=[current_role("Senior ML Engineer", "2023-01")],
            github_url="https://github.com/strong",
        ),
        # skills .7*2/3 + .3*1 = .766667 | traj lateral .5 | recency 27mo ago → .75
        # behav .4+.2+.2*1.0("open to")+.2*.5 = .90 | seniority 4y → 1.0
        # .306667 + .10 + .1125 + .135 + .10 = 75.42
        make_candidate(
            "C_GAP",
            skills=["Python", "Docker", "Kubernetes"],
            trajectory_label="lateral",
            total_years_exp=4,
            experience=[past_role("ML Engineer", "2020-01", "2024-03")],
            raw_text="Open to new opportunities in ML.",
            github_url="https://github.com/gap",
        ),
        # skills .7*2/3 = .466667 | traj steady_climb .7 | recency 1.0
        # behav .4*.75+0+.2*.6(9mo)+.2 = .62 | seniority 5y → 1.0
        # .186667 + .14 + .15 + .093 + .10 = 66.97
        make_candidate(
            "C_GOOD",
            skills=["Python", "PyTorch", "SQL"],
            trajectory_label="steady_climb",
            total_years_exp=5,
            experience=[current_role("ML Engineer", "2025-09")],
            fields_present=["skills", "experience", "education"],
            fields_missing=["location"],
            confidence_level="MEDIUM",
        ),
        # skills .7 | traj steady .7 | recency 1.0
        # behav .4+0+.2*.3(>60mo)+.2 = .66 | seniority 18y: 1 - 8*.15 → 0
        # .28 + .14 + .15 + .099 + 0 = 66.90  (overqualification drops it below C_GOOD)
        make_candidate(
            "C_OVER",
            skills=["Python", "PyTorch", "Docker"],
            trajectory_label="steady",
            total_years_exp=18,
            experience=[current_role("Principal Engineer", "2015-01")],
        ),
        # skills 0 | traj declining .1 | recency 1.0
        # behav .4+0+.2*.5(48mo)+.2 = .70 | seniority 7y → 1.0
        # 0 + .02 + .15 + .105 + .10 = 37.50
        make_candidate(
            "C_POOR",
            skills=["Excel", "Photoshop"],
            trajectory_label="declining",
            total_years_exp=7,
            experience=[current_role("Graphic Designer", "2022-06")],
        ),
        # skills .7*1/3 = .233333 | traj insufficient_data .4 | recency no exp → .5
        # behav .4*.5+0+.2*.5+.2*.5 = .40 | seniority 0y: 1 - 4*.2 = .2
        # .093333 + .08 + .075 + .06 + .02 = 32.83
        make_candidate(
            "C_FRESH",
            skills=["Python"],
            trajectory_label="insufficient_data",
            total_years_exp=0,
            is_fresher=True,
            fields_present=["skills", "education"],
            fields_missing=["experience", "location"],
            confidence_level="LOW",
        ),
    ]


EXPECTED_ORDER = ["C_STRONG", "C_GAP", "C_GOOD", "C_OVER", "C_POOR", "C_FRESH"]

EXPECTED_SCORES = {
    "C_STRONG": 98.50,
    "C_GAP": 75.42,
    "C_GOOD": 66.97,
    "C_OVER": 66.90,
    "C_POOR": 37.50,
    "C_FRESH": 32.83,
}

EXPECTED_TIERS = {
    "C_STRONG": "strong_match",
    "C_GAP": "strong_match",
    "C_GOOD": "strong_match",
    "C_OVER": "strong_match",
    "C_POOR": "weak_match",
    "C_FRESH": "weak_match",
}


def _rank(candidates, jd=None):
    """Run Layer 2 end to end with the fake embedding model."""
    return run_layer2(jd or make_jd(), candidates, FakeModel(), top_k=50)


@pytest.fixture(autouse=True)
def _frozen(frozen_layer2_clock):
    """All tests in this module run with the Layer 2 clock frozen."""
    return frozen_layer2_clock


class TestFixedRanking:
    """A fixed input must produce a fixed, hand-verified ranking."""

    def test_ranking_order(self):
        shortlist, _, _ = _rank(_pool())
        assert [s.candidate_id for s in shortlist] == EXPECTED_ORDER

    def test_ranks_are_one_indexed_and_contiguous(self):
        shortlist, _, _ = _rank(_pool())
        assert [s.rank for s in shortlist] == list(range(1, len(EXPECTED_ORDER) + 1))

    @pytest.mark.parametrize("cid,expected", sorted(EXPECTED_SCORES.items()))
    def test_fusion_score(self, cid, expected):
        shortlist, _, _ = _rank(_pool())
        by_id = {s.candidate_id: s for s in shortlist}
        assert by_id[cid].signal_fusion_score == pytest.approx(expected, abs=0.01)
        # Before GitHub enrichment / LLM, final_score is the fusion score.
        assert by_id[cid].final_score == by_id[cid].signal_fusion_score

    def test_tiers(self):
        shortlist, _, _ = _rank(_pool())
        assert {s.candidate_id: s.tier for s in shortlist} == EXPECTED_TIERS

    def test_no_low_match_warning_and_no_hidden_gems(self):
        _, gems, low_match = _rank(_pool())
        assert low_match is False
        assert gems == []  # every candidate is already in the top 10

    def test_skills_matched_and_missing(self):
        shortlist, _, _ = _rank(_pool())
        by_id = {s.candidate_id: s for s in shortlist}
        assert by_id["C_STRONG"].skills_matched == ["Python", "PyTorch", "Docker"]
        assert by_id["C_STRONG"].skills_missing == []
        assert by_id["C_GOOD"].skills_matched == ["Python", "PyTorch"]
        assert by_id["C_GOOD"].skills_missing == ["Docker"]
        assert by_id["C_POOR"].skills_missing == ["Python", "PyTorch", "Docker"]


class TestDeterminism:
    """Same inputs → same outputs, regardless of order or repetition."""

    def test_repeated_runs_are_identical(self):
        first, _, _ = _rank(_pool())
        second, _, _ = _rank(_pool())
        assert [dataclasses.asdict(s) for s in first] == [
            dataclasses.asdict(s) for s in second
        ]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_input_order_does_not_change_ranking(self, seed):
        pool = _pool()
        random.Random(seed).shuffle(pool)
        shortlist, _, _ = _rank(pool)
        assert [s.candidate_id for s in shortlist] == EXPECTED_ORDER


class TestFusionContract:
    """Score normalization contract from CLAUDE.md."""

    def test_signals_normalized_and_weighted_sum_to_final(self):
        shortlist, _, _ = _rank(_pool())
        for sc in shortlist:
            total = 0.0
            for name, bd in sc.signal_breakdown.items():
                assert 0.0 <= bd.normalized <= 1.0, (sc.candidate_id, name)
                assert bd.weighted == pytest.approx(
                    bd.normalized * DEFAULT_TEST_WEIGHTS[name]
                )
                total += bd.weighted
            assert total * 100 == pytest.approx(sc.signal_fusion_score, abs=0.01)

    def test_overqualified_is_flagged(self):
        shortlist, _, _ = _rank(_pool())
        by_id = {s.candidate_id: s for s in shortlist}
        assert by_id["C_OVER"].flags.is_overqualified is True
        assert by_id["C_OVER"].seniority_label == "overqualified"
        assert by_id["C_OVER"].signal_breakdown["seniority"].normalized == 0.0
        assert not any(
            by_id[c].flags.is_overqualified for c in EXPECTED_ORDER if c != "C_OVER"
        )

    @staticmethod
    def _over_perfect():
        """Perfect skills/trajectory, but 20 years against a 4–10 year band (88.50)."""
        return make_candidate(
            "C_OVER_PERFECT",
            skills=["Python", "PyTorch", "Docker", "Kubernetes"],
            trajectory_label="high_growth",
            total_years_exp=20,
            experience=[current_role("Principal Engineer", "2023-01")],
            github_url="https://github.com/over",
        )

    def test_overqualified_loses_to_aligned_peer(self):
        """The seniority penalty costs an otherwise-identical profile the top spot."""
        shortlist, _, _ = _rank(_pool() + [self._over_perfect()])
        assert [s.candidate_id for s in shortlist[:2]] == ["C_STRONG", "C_OVER_PERFECT"]

    @pytest.mark.xfail(
        strict=True,
        reason="Spec gap: CLAUDE.md says an overqualified candidate must never be "
               "ranked #1, but run_layer2 only applies a score penalty.",
    )
    def test_overqualified_never_ranked_first(self):
        weaker = [c for c in _pool() if c.candidate_id in {"C_GOOD", "C_POOR"}]
        shortlist, _, _ = _rank(weaker + [self._over_perfect()])
        assert shortlist[0].candidate_id != "C_OVER_PERFECT"

    def test_jd_weights_drive_ranking(self):
        """Putting all weight on seniority reorders the pool accordingly."""
        jd = make_jd(weights={
            "skills": 0.0, "trajectory": 0.0, "recency": 0.0,
            "behavioral": 0.0, "seniority": 1.0,
        })
        shortlist, _, _ = _rank(_pool(), jd=jd)
        by_id = {s.candidate_id: s.signal_fusion_score for s in shortlist}
        assert by_id["C_STRONG"] == by_id["C_POOR"] == 100.0
        assert by_id["C_FRESH"] == pytest.approx(20.0)
        assert by_id["C_OVER"] == 0.0
        assert shortlist[-1].candidate_id == "C_OVER"

    def test_low_match_warning_when_pool_is_weak(self):
        weak = [c for c in _pool() if c.candidate_id in {"C_POOR", "C_FRESH"}]
        shortlist, _, low_match = _rank(weak)
        assert shortlist[0].signal_fusion_score < 40
        assert low_match is True

    def test_empty_pool(self):
        assert run_layer2(make_jd(), [], FakeModel()) == ([], [], True)

    def test_fuse_signals_single_candidate_matches_run_layer2(self):
        jd = make_jd()
        pool = _pool()
        emb = precompute_skill_embeddings(jd, pool, FakeModel())
        sc, score = fuse_signals(pool[0], jd, emb)
        assert sc.rank == 0  # caller assigns ranks
        assert score == pytest.approx(EXPECTED_SCORES["C_STRONG"], abs=0.01)
