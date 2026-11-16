"""
Tests for the hackathon submission ranker (rank.py).

rank.py fuses four signals (skills .45, experience .20, behavioral .25,
location .10) into a raw score and writes the top-k as submission.csv.
The clock is frozen at 2026-06-01 so activity-recency is reproducible.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import date

import pytest

import rank

TODAY = date(2026, 6, 1)


class FrozenDate(date):
    """date subclass whose today() always returns TODAY."""

    @classmethod
    def today(cls):
        """Return the frozen date."""
        return TODAY


@pytest.fixture(autouse=True)
def _frozen_today(monkeypatch):
    """rank.py calls date.today() in main() and generate_reasoning()."""
    monkeypatch.setattr(rank, "date", FrozenDate)


def _signals(**overrides):
    """A fully populated redrob_signals block."""
    base = {
        "last_active_date": "2026-05-20",
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 16.8,
        "interview_completion_rate": 0.9,
        "github_activity_score": 75,
        "saved_by_recruiters_30d": 5,
        "profile_completeness_score": 90,
        "skill_assessment_scores": {"FAISS": 80},
        "notice_period_days": 30,
        "willing_to_relocate": False,
    }
    base.update(overrides)
    return base


R_TOP = {
    "candidate_id": "R_TOP",
    "profile": {"current_title": "ML Engineer", "years_of_experience": 7,
                "location": "Bangalore", "country": "India"},
    "skills": [
        {"name": "FAISS", "proficiency_level": "expert", "endorsements": 99, "duration_months": 60},
        {"name": "RAG", "proficiency_level": "advanced", "endorsements": 0, "duration_months": 0},
        {"name": "PyTorch", "proficiency_level": "intermediate", "endorsements": 0, "duration_months": 30},
        {"name": "Excel", "proficiency_level": "expert", "endorsements": 50, "duration_months": 60},
    ],
    "redrob_signals": _signals(),
}

# Same profile with a non-AI title → skills score × 0.1.
R_BADTITLE = {**R_TOP, "candidate_id": "R_BADTITLE",
              "profile": {**R_TOP["profile"], "current_title": "Software Engineer"}}

R_REMOTE = {
    "candidate_id": "R_REMOTE",
    "profile": {"current_title": "Research Engineer", "years_of_experience": 12,
                "location": "Berlin", "country": "Germany"},
    "skills": [
        {"name": "PyTorch", "proficiency_level": "expert", "endorsements": 40, "duration_months": 48},
        {"name": "Deep Learning", "proficiency_level": "advanced", "endorsements": 20, "duration_months": 60},
        {"name": "Embeddings", "proficiency_level": "expert", "endorsements": 10, "duration_months": 24},
    ],
    "redrob_signals": _signals(last_active_date="2026-02-01", open_to_work_flag=False,
                               recruiter_response_rate=0.5, willing_to_relocate=True,
                               skill_assessment_scores={}),
}

R_MID = {
    "candidate_id": "R_MID",
    "profile": {"current_title": "Data Scientist", "years_of_experience": 4,
                "location": "Pune", "country": "India"},
    "skills": [
        {"name": "NLP", "proficiency_level": "advanced", "endorsements": 9, "duration_months": 24},
        {"name": "Hugging Face", "proficiency_level": "intermediate", "endorsements": 3, "duration_months": 12},
    ],
    "redrob_signals": _signals(github_activity_score=20, saved_by_recruiters_30d=1,
                               skill_assessment_scores={}),
}

R_POOR = {
    "candidate_id": "R_POOR",
    "profile": {"current_title": "Accountant", "years_of_experience": 10,
                "location": "Chennai", "country": "India"},
    "skills": [{"name": "Tally", "proficiency_level": "expert"},
               {"name": "Excel", "proficiency_level": "expert"}],
    "redrob_signals": _signals(last_active_date="2025-01-01", open_to_work_flag=False,
                               recruiter_response_rate=0.1, github_activity_score=-1,
                               interview_completion_rate=0.3, skill_assessment_scores={}),
}

R_EMPTY = {"candidate_id": "R_EMPTY"}

POOL = [R_POOR, R_MID, R_EMPTY, R_BADTITLE, R_REMOTE, R_TOP]


def _run_main(monkeypatch, tmp_path, pool, top_k, name="submission.csv"):
    """Write pool as JSONL, run rank.main() and return the parsed CSV rows."""
    jsonl = tmp_path / f"{name}.jsonl"
    jsonl.write_text("".join(json.dumps(c) + "\n" for c in pool), encoding="utf-8")
    out = tmp_path / name
    monkeypatch.setattr(sys, "argv", [
        "rank.py", "--candidates", str(jsonl), "--jd", "unused.docx",
        "--out", str(out), "--top-k", str(top_k),
    ])
    assert rank.main() == 0
    with out.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh)), out


class TestScoreCandidate:
    """Per-signal scores for a hand-computed candidate."""

    def test_hand_computed_scores(self):
        """
        skills:  FAISS 1.0*(1+.2*1+.1*1)=1.3, RAG .8, PyTorch .6*(1+.1*.5)=.63
                 + assessment 80/100*.1=.08 → 2.81/4 = .7025, +.03 (3 matched) = .7325
        experience: 7y in [5, 9] → 1.0
        behavioral: .25*1 + .2*1 + .2*(.8*(1-16.8/168)) + .15*.9 + .1*1 + .05*.5 + .05*.9 = .899
        location: Bangalore → 1.0
        raw: .45*.7325 + .2 + .25*.899 + .1 = .854375
        """
        s = rank.score_candidate(R_TOP, TODAY)
        assert s["skills"] == pytest.approx(0.7325)
        assert s["experience"] == 1.0
        assert s["behavioral"] == pytest.approx(0.899)
        assert s["location"] == 1.0
        assert s["raw_score"] == pytest.approx(0.854375)

    def test_bad_title_penalty_is_tenfold(self):
        good = rank.score_candidate(R_TOP, TODAY)
        bad = rank.score_candidate(R_BADTITLE, TODAY)
        assert bad["skills"] == pytest.approx(good["skills"] * 0.1)
        for key in ("experience", "behavioral", "location"):
            assert bad[key] == good[key]

    def test_empty_record_does_not_crash_and_stays_in_bounds(self):
        s = rank.score_candidate(R_EMPTY, TODAY)
        assert s["skills"] == 0.0
        assert 0.0 <= s["raw_score"] <= 1.0

    @pytest.mark.parametrize("years,expected", [
        (0, 0.20), (2.5, 0.40), (3.5, 0.60), (4.5, 0.80), (5, 1.0), (9, 1.0),
        (10, 0.75), (12, 0.55), (20, 0.35),
    ])
    def test_experience_band(self, years, expected):
        assert rank._score_experience({"profile": {"years_of_experience": years}}) == expected

    @pytest.mark.parametrize("profile,signals,expected", [
        ({"location": "Noida, UP"}, {}, 1.0),
        ({"location": "Kochi", "country": "India"}, {}, 0.8),
        ({"location": "Berlin"}, {"willing_to_relocate": True}, 0.6),
        ({"location": "Berlin"}, {}, 0.3),
        ({}, {}, 0.3),
    ])
    def test_location_is_boost_only(self, profile, signals, expected):
        cand = {"profile": profile, "redrob_signals": signals}
        assert rank._score_location(cand) == expected

    def test_all_scores_bounded(self):
        for cand in POOL:
            for key, value in rank.score_candidate(cand, TODAY).items():
                assert 0.0 <= value <= 1.0, (cand["candidate_id"], key, value)


class TestFixedRanking:
    """End-to-end: a fixed JSONL in → a fixed submission.csv out."""

    # Raw scores: R_TOP .854, R_REMOTE .713, R_MID .656, R_BADTITLE .558,
    # R_POOR .323, R_EMPTY .110 — the bad-title penalty drops an otherwise
    # identical-to-R_TOP profile below a weaker but correctly-titled one.
    EXPECTED_ORDER = ["R_TOP", "R_REMOTE", "R_MID", "R_BADTITLE", "R_POOR", "R_EMPTY"]

    def test_ranking_order(self, monkeypatch, tmp_path):
        rows, _ = _run_main(monkeypatch, tmp_path, POOL, top_k=6)
        assert [r["candidate_id"] for r in rows] == self.EXPECTED_ORDER
        assert [int(r["rank"]) for r in rows] == [1, 2, 3, 4, 5, 6]

    def test_top_score_matches_hand_computed(self, monkeypatch, tmp_path):
        rows, _ = _run_main(monkeypatch, tmp_path, POOL, top_k=6)
        assert float(rows[0]["score"]) == pytest.approx(0.854375, abs=1e-6)

    def test_output_is_byte_identical_across_runs(self, monkeypatch, tmp_path):
        _, first = _run_main(monkeypatch, tmp_path, POOL, top_k=6, name="a.csv")
        _, second = _run_main(monkeypatch, tmp_path, POOL, top_k=6, name="b.csv")
        assert first.read_bytes() == second.read_bytes()

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_input_order_does_not_change_ranking(self, monkeypatch, tmp_path, seed):
        pool = list(POOL)
        random.Random(seed).shuffle(pool)
        rows, _ = _run_main(monkeypatch, tmp_path, pool, top_k=6)
        assert [r["candidate_id"] for r in rows] == self.EXPECTED_ORDER

    def test_top_k_truncates(self, monkeypatch, tmp_path):
        rows, _ = _run_main(monkeypatch, tmp_path, POOL, top_k=3)
        assert [r["candidate_id"] for r in rows] == self.EXPECTED_ORDER[:3]

    def test_output_passes_validation(self, monkeypatch, tmp_path):
        rows, _ = _run_main(monkeypatch, tmp_path, POOL, top_k=6)
        assert rank.validate_submission(rows, top_k=6) == []
        assert all(0 < len(r["reasoning"]) <= 200 for r in rows)

    def test_blank_and_malformed_lines_are_skipped(self, monkeypatch, tmp_path):
        jsonl = tmp_path / "messy.jsonl"
        jsonl.write_text(
            json.dumps(R_TOP) + "\n\n{broken json\n" + json.dumps(R_MID) + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.csv"
        monkeypatch.setattr(sys, "argv", [
            "rank.py", "--candidates", str(jsonl), "--jd", "x",
            "--out", str(out), "--top-k", "2",
        ])
        assert rank.main() == 0
        with out.open(encoding="utf-8") as fh:
            assert [r["candidate_id"] for r in csv.DictReader(fh)] == ["R_TOP", "R_MID"]

    def test_fails_when_fewer_candidates_than_top_k(self, monkeypatch, tmp_path):
        jsonl = tmp_path / "small.jsonl"
        jsonl.write_text(json.dumps(R_TOP) + "\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "rank.py", "--candidates", str(jsonl), "--jd", "x",
            "--out", str(tmp_path / "o.csv"), "--top-k", "5",
        ])
        assert rank.main() == 1


class TestValidateSubmission:
    """validate_submission catches every malformed-row class it documents."""

    @staticmethod
    def _rows(n=3):
        return [{"candidate_id": f"C{i}", "rank": i, "score": 1.0 - i / 10,
                 "reasoning": "ok"} for i in range(1, n + 1)]

    def test_valid(self):
        assert rank.validate_submission(self._rows(), 3) == []

    @pytest.mark.parametrize("mutate,needle", [
        (lambda r: r.pop(), "Expected 3 rows"),
        (lambda r: r[1].update(rank=1), "used more than once"),
        (lambda r: r[2].update(rank=9), "out of range"),
        (lambda r: r[0].update(score=1.5), "outside [0.0, 1.0]"),
        (lambda r: r[2].update(score=0.95), "not non-increasing"),
        (lambda r: r[0].update(candidate_id=""), "empty candidate_id"),
        (lambda r: r[1].update(reasoning="  "), "empty reasoning"),
        (lambda r: r[0].update(rank="first"), "not an integer"),
        (lambda r: r[0].update(score=math.nan), "outside [0.0, 1.0]"),
    ])
    def test_invalid(self, mutate, needle):
        rows = self._rows()
        mutate(rows)
        errors = rank.validate_submission(rows, 3)
        assert any(needle in e for e in errors), errors
