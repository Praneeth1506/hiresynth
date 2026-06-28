"""
rank.py — HireSynth hackathon submission entry point.

Ranks up to 100,000 candidates against a Senior AI Engineer JD
in under 5 minutes on CPU with zero external API calls.

PIPELINE (no LLM, no API):
  1. Load JD skills as a fixed list (hardcoded from the JD spec).
  2. Stream candidates.jsonl line by line (constant memory).
  3. Score each candidate across four signals:
       A. Skills match        (40%)
       B. Experience fit      (20%)
       C. Behavioral signals  (30%)
       D. Location/relocation (10%)
  4. Min-max normalize scores across all candidates.
  5. Take top-k (default 100).
  6. Generate rule-based reasoning per candidate.
  7. Validate output, then write submission.csv.

Usage:
    python rank.py --candidates candidates.jsonl --jd job_description.docx --out submission.csv
    python rank.py --candidates candidates.jsonl --jd job_description.docx --out submission.csv --sample 5000
    python rank.py --candidates candidates.jsonl --jd job_description.docx --out submission.csv --top-k 50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

np.random.seed(42)

# ---------------------------------------------------------------------------
# JD specification — hardcoded from the Senior AI Engineer @ Redrob JD
# ---------------------------------------------------------------------------

JD_CORE_SKILLS: list[str] = [
    "sentence-transformers", "Sentence Transformers",
    "embeddings", "Embeddings",
    "FAISS", "Milvus", "Qdrant",
    "vector database", "hybrid search",
    "Elasticsearch", "OpenSearch",
    "Weaviate", "Pinecone",
    "RAG", "retrieval augmented",
    "Information Retrieval",
    "Hugging Face", "HuggingFace",
    "MLOps", "LLM", "LLMs",
    "fine-tuning", "Fine-tuning LLMs",
    "NDCG", "MRR", "evaluation framework",
    "vector search", "semantic search",
    "reranking", "re-ranking",
    "BM25", "NLP",
    "PyTorch", "TensorFlow",
    "deep learning",
]

# Lowercase version computed once for O(1) matching
_JD_SKILLS_LOWER: list[str] = [s.lower() for s in JD_CORE_SKILLS]

# Proficiency → weight mapping
_PROFICIENCY_WEIGHTS: dict[str, float] = {
    "expert":        1.0,
    "advanced":      0.8,
    "intermediate":  0.6,
    "beginner":      0.3,
}

# Target cities for location scoring
_TARGET_CITIES: list[str] = [
    "pune", "noida", "bangalore", "bengaluru",
    "delhi", "mumbai", "hyderabad", "chennai",
]

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

_BAD_TITLES: tuple[str, ...] = (
    "civil engineer", "mechanical engineer", "accountant",
    "graphic designer", "customer support", "marketing manager",
    "content writer", "sales", "hr manager",
    "operations manager", "project manager", "business analyst",
)


def _score_skills(candidate: dict[str, Any], current_title: str = "") -> float:
    """
    Compute the skills match score (0–1) using JD_CORE_SKILLS.

    Each matched skill contributes a weighted score based on proficiency,
    endorsement count (log-scaled), and duration.  An additional small boost
    is applied for platform-assessed skills.  The raw sum is normalized by
    dividing by 5 (expected matched skills for a perfect score), capped at 1.0.

    Title penalty: non-AI titles (civil engineer, accountant, etc.) indicate
    the candidate's AI skills are incidental.  A 0.4x multiplier is applied
    to the final score for these titles.
    """
    skills: list[dict] = candidate.get("skills") or []
    redrob: dict = candidate.get("redrob_signals") or {}
    assessments: dict[str, float] = {}

    # Build assessment lookup from redrob_signals.skill_assessment_scores
    raw_assessments = redrob.get("skill_assessment_scores") or {}
    if isinstance(raw_assessments, dict):
        for skill_name, score_val in raw_assessments.items():
            try:
                assessments[skill_name.lower()] = float(score_val)
            except (TypeError, ValueError):
                pass

    total_skill_score = 0.0

    for skill_entry in skills:
        if not isinstance(skill_entry, dict):
            continue

        skill_name: str = str(skill_entry.get("name") or "").lower()
        if not skill_name:
            continue

        # Check if this skill matches any JD core skill (substring, both directions)
        matched = any(
            jd_skill in skill_name or skill_name in jd_skill
            for jd_skill in _JD_SKILLS_LOWER
        )
        if not matched:
            continue

        # Proficiency weight
        proficiency: str = str(skill_entry.get("proficiency_level") or "").lower()
        prof_weight: float = _PROFICIENCY_WEIGHTS.get(proficiency, 0.5)

        # Endorsement boost: log(1 + n) / log(100)
        try:
            endorsements = float(skill_entry.get("endorsements") or 0)
        except (TypeError, ValueError):
            endorsements = 0.0
        endorsement_boost = math.log1p(endorsements) / math.log(100)

        # Duration boost: min(months, 60) / 60
        try:
            duration_months = float(skill_entry.get("duration_months") or 0)
        except (TypeError, ValueError):
            duration_months = 0.0
        duration_boost = min(duration_months, 60.0) / 60.0

        skill_score = prof_weight * (1.0 + 0.2 * endorsement_boost + 0.1 * duration_boost)
        total_skill_score += skill_score

    # Assessment boost: small verified signal for each JD-matched assessment
    for assessed_skill, assessed_score in assessments.items():
        if any(
            jd_skill in assessed_skill or assessed_skill in jd_skill
            for jd_skill in _JD_SKILLS_LOWER
        ):
            try:
                total_skill_score += float(assessed_score) / 100.0 * 0.1
            except (TypeError, ValueError):
                pass

    score = min(total_skill_score / 4.0, 1.0)

    title_lower = current_title.lower()
    if any(t in title_lower for t in _BAD_TITLES):
        score *= 0.4

    return score


def _score_experience(candidate: dict[str, Any]) -> float:
    """
    Score experience fit (0–1) based on years_of_experience vs the target 5–9 year band.

    Perfect band (5–9 yrs) → 1.0
    Overqualified (>9 yrs)  → decays from 1.0 toward 0.5
    Near-miss (3–5 yrs)     → 0.7
    Junior (<3 yrs)         → 0.3
    """
    profile: dict = candidate.get("profile") or {}
    try:
        years = float(profile.get("years_of_experience") or 0)
    except (TypeError, ValueError):
        years = 0.0

    if 5.0 <= years <= 9.0:    return 1.00
    if 4.0 <= years < 5.0:     return 0.80
    if 9.0 < years <= 11.0:    return 0.75
    if 3.0 <= years < 4.0:     return 0.60
    if 11.0 < years <= 14.0:   return 0.55
    if years > 14.0:            return 0.35
    if 2.0 <= years < 3.0:     return 0.40
    return 0.20


def _score_behavioral(candidate: dict[str, Any], today: date) -> float:
    """
    Compute the behavioral signals score (0–1) from redrob_signals fields.

    Sub-signals and weights:
      activity_score    0.25  — recency of last platform activity
      openness          0.20  — open_to_work_flag
      response_quality  0.20  — recruiter_response_rate × time-decay
      reliability       0.15  — interview_completion_rate
      github_score      0.10  — github_activity_score (−1 = no profile)
      saved             0.05  — saved_by_recruiters_30d (capped at 10)
      completeness      0.05  — profile_completeness_score / 100
    """
    redrob: dict = candidate.get("redrob_signals") or {}

    # Activity recency
    activity = 0.2
    last_active_raw = redrob.get("last_active_date")
    if last_active_raw:
        try:
            last_active = date.fromisoformat(str(last_active_raw))
            days_since = (today - last_active).days
            if days_since <= 30:
                activity = 1.0
            elif days_since <= 90:
                activity = 0.8
            elif days_since <= 180:
                activity = 0.5
            else:
                activity = 0.2
        except (ValueError, TypeError):
            pass

    # Openness
    openness = 1.0 if redrob.get("open_to_work_flag") else 0.4

    # Response quality
    try:
        rr = float(redrob.get("recruiter_response_rate") or 0)
    except (TypeError, ValueError):
        rr = 0.0
    try:
        rt = float(redrob.get("avg_response_time_hours") or 0)
    except (TypeError, ValueError):
        rt = 0.0
    response_quality = rr * max(0.0, 1.0 - rt / 168.0)

    # Reliability
    try:
        reliability = float(redrob.get("interview_completion_rate") or 0)
    except (TypeError, ValueError):
        reliability = 0.0

    # GitHub
    try:
        gas = float(redrob.get("github_activity_score") or 0)
    except (TypeError, ValueError):
        gas = -1.0
    if gas == -1.0:
        github_score = 0.3
    elif gas >= 70.0:
        github_score = 1.0
    elif gas >= 40.0:
        github_score = 0.7
    elif gas >= 10.0:
        github_score = 0.5
    else:
        github_score = 0.3

    # Saved by recruiters
    try:
        saved_count = float(redrob.get("saved_by_recruiters_30d") or 0)
    except (TypeError, ValueError):
        saved_count = 0.0
    saved = min(saved_count / 10.0, 1.0)

    # Profile completeness
    try:
        completeness = float(redrob.get("profile_completeness_score") or 0) / 100.0
    except (TypeError, ValueError):
        completeness = 0.0

    return (
        0.25 * activity
        + 0.20 * openness
        + 0.20 * response_quality
        + 0.15 * reliability
        + 0.10 * github_score
        + 0.05 * saved
        + 0.05 * completeness
    )


def _score_location(candidate: dict[str, Any]) -> float:
    """
    Score location fit (0–1) for the Bangalore/Pune/Noida/India target.

    Exact city match → 1.0
    India but not target city → 0.8
    Willing to relocate → 0.6
    None of the above → 0.3

    Location is a soft boost: candidates are never penalized below 0.3.
    """
    profile: dict = candidate.get("profile") or {}
    redrob: dict = candidate.get("redrob_signals") or {}

    location: str = str(profile.get("location") or "").lower()
    country: str = str(profile.get("country") or "").lower()
    willing: bool = bool(redrob.get("willing_to_relocate"))

    if any(city in location for city in _TARGET_CITIES):
        return 1.0
    if "india" in country or "india" in location:
        return 0.8
    if willing:
        return 0.6
    return 0.3


def score_candidate(candidate: dict[str, Any], today: date) -> dict[str, float]:
    """
    Compute all four signal scores and the raw weighted total for one candidate.

    Returns a dict with keys: skills, experience, behavioral, location, raw_score.
    """
    current_title: str = str((candidate.get("profile") or {}).get("current_title") or "")
    skills = _score_skills(candidate, current_title=current_title)
    experience = _score_experience(candidate)
    behavioral = _score_behavioral(candidate, today)
    location = _score_location(candidate)

    raw = (
        0.45 * skills
        + 0.20 * experience
        + 0.25 * behavioral
        + 0.10 * location
    )

    return {
        "skills": skills,
        "experience": experience,
        "behavioral": behavioral,
        "location": location,
        "raw_score": raw,
    }


# ---------------------------------------------------------------------------
# Rule-based reasoning generator
# ---------------------------------------------------------------------------

def generate_reasoning(candidate: dict[str, Any], scores: dict[str, float]) -> str:
    """
    Produce a one-sentence reasoning string using only candidate data (no API).

    Format: "<title> with <years> yrs; <N> matched AI/ML skills; response rate <X%>
             [; actively looking] [; GitHub score <N>] [; saved by <N> recruiters]
             [; concern: <...>]."
    """
    profile: dict = candidate.get("profile") or {}
    redrob: dict = candidate.get("redrob_signals") or {}
    skills_list: list[dict] = candidate.get("skills") or []

    title: str = str(profile.get("current_title") or "Unknown Role")
    try:
        years: float = float(profile.get("years_of_experience") or 0)
    except (TypeError, ValueError):
        years = 0.0

    # Matched skills (name list, up to 3)
    cand_skill_names: list[str] = [
        str(s.get("name") or "") for s in skills_list if isinstance(s, dict)
    ]
    matched: list[str] = [
        s for s in cand_skill_names
        if any(j in s.lower() or s.lower() in j for j in _JD_SKILLS_LOWER)
    ][:3]

    try:
        rr: float = float(redrob.get("recruiter_response_rate") or 0)
    except (TypeError, ValueError):
        rr = 0.0

    parts: list[str] = [
        f"{title} with {years:.1f} yrs",
        (
            f"{len(matched)} matched AI/ML skills ({', '.join(matched)})"
            if matched
            else "limited direct skill match"
        ),
        f"response rate {rr:.0%}",
    ]

    # Positive signals
    if redrob.get("open_to_work_flag"):
        parts.append("actively looking")

    try:
        gas: float = float(redrob.get("github_activity_score") or -1)
    except (TypeError, ValueError):
        gas = -1.0
    if gas >= 50.0:
        parts.append(f"GitHub score {gas:.0f}")

    try:
        saved_count: float = float(redrob.get("saved_by_recruiters_30d") or 0)
    except (TypeError, ValueError):
        saved_count = 0.0
    if saved_count >= 5:
        parts.append(f"saved by {saved_count:.0f} recruiters")

    # Concerns
    concerns: list[str] = []

    last_active_raw = redrob.get("last_active_date")
    if last_active_raw:
        try:
            days_since = (date.today() - date.fromisoformat(str(last_active_raw))).days
            if days_since > 180:
                concerns.append(f"inactive {days_since // 30}mo")
        except (ValueError, TypeError):
            pass

    try:
        notice: float = float(redrob.get("notice_period_days") or 0)
    except (TypeError, ValueError):
        notice = 0.0
    if notice > 90:
        concerns.append(f"notice {notice:.0f}d")

    if concerns:
        parts.append("concern: " + ", ".join(concerns))

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_submission(rows: list[dict[str, Any]], top_k: int) -> list[str]:
    """
    Validate the submission rows before writing.

    Checks (matching what validate_submission.py would enforce):
      - Exactly top_k rows
      - Ranks are integers 1..top_k, each used exactly once
      - Scores are floats in [0.0, 1.0]
      - Scores are non-increasing by rank
      - candidate_id is non-empty for every row
      - reasoning is non-empty for every row

    Returns a list of error strings (empty list = valid).
    """
    errors: list[str] = []

    if len(rows) != top_k:
        errors.append(f"Expected {top_k} rows, got {len(rows)}")

    ranks_seen: set[int] = set()
    prev_score: float = 1.1

    for i, row in enumerate(rows):
        cid = str(row.get("candidate_id") or "").strip()
        if not cid:
            errors.append(f"Row {i + 1}: empty candidate_id")

        try:
            rank = int(row["rank"])
        except (KeyError, ValueError, TypeError):
            errors.append(f"Row {i + 1}: rank is not an integer")
            continue

        if rank < 1 or rank > top_k:
            errors.append(f"Row {i + 1}: rank {rank} out of range [1, {top_k}]")
        if rank in ranks_seen:
            errors.append(f"Row {i + 1}: rank {rank} used more than once")
        ranks_seen.add(rank)

        try:
            score = float(row["score"])
        except (KeyError, ValueError, TypeError):
            errors.append(f"Row {i + 1}: score is not a float")
            continue

        if not (0.0 <= score <= 1.0):
            errors.append(f"Row {i + 1}: score {score} outside [0.0, 1.0]")
        if score > prev_score + 1e-9:
            errors.append(
                f"Row {i + 1}: score {score:.6f} > previous {prev_score:.6f} (not non-increasing)"
            )
        prev_score = score

        reasoning = str(row.get("reasoning") or "").strip()
        if not reasoning:
            errors.append(f"Row {i + 1}: empty reasoning")

    expected_ranks = set(range(1, top_k + 1))
    missing = expected_ranks - ranks_seen
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)[:10]}")

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="rank.py",
        description="HireSynth — hackathon submission ranker (no API calls, CPU only)",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl (one JSON object per line)",
    )
    parser.add_argument(
        "--jd",
        required=True,
        help="Path to job description file (docx or txt; used only for record-keeping — JD skills are hardcoded)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path (e.g. submission.csv)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        dest="top_k",
        help="Number of candidates to output (default: 100)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        dest="sample",
        help="Only process the first N candidates (for testing; 0 = all)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the full ranking pipeline end-to-end.

    Returns 0 on success, 1 on validation failure.
    """
    args = _parse_args()
    wall_start = time.perf_counter()

    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    top_k: int = args.top_k
    sample: int = args.sample

    if not candidates_path.exists():
        print(f"[ERROR] Candidates file not found: {candidates_path}", file=sys.stderr)
        return 1

    today = date.today()

    # ------------------------------------------------------------------
    # Step 2 — stream and score all candidates
    # ------------------------------------------------------------------
    print(f"\n  Loading and scoring candidates from {candidates_path.name} ...")

    all_scores: list[tuple[str, dict[str, float], dict[str, Any]]] = []
    # (candidate_id, scores_dict, candidate_record)

    with candidates_path.open(encoding="utf-8") as fh:
        line_iter = tqdm(fh, desc="  Scoring", unit="cand", mininterval=0.5)
        for line_num, line in enumerate(line_iter, start=1):
            if sample and line_num > sample:
                break
            line = line.strip()
            if not line:
                continue
            try:
                candidate: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid: str = str(
                candidate.get("candidate_id")
                or candidate.get("id")
                or f"UNKNOWN_{line_num}"
            )
            scores = score_candidate(candidate, today)
            all_scores.append((cid, scores, candidate))

    total_candidates = len(all_scores)
    print(f"  Scored {total_candidates:,} candidates.")

    if total_candidates == 0:
        print("[ERROR] No candidates loaded — check the input file.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 4 — sort by raw score directly (already bounded 0–1)
    # ------------------------------------------------------------------
    # Use raw scores directly — already bounded 0-1
    scored = [(cid, s["raw_score"], r) for cid, s, r in all_scores]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_entries = scored[:top_k]

    # ------------------------------------------------------------------
    # Steps 5 & 6 — generate reasoning and build output rows
    # ------------------------------------------------------------------
    print(f"  Generating reasoning for top {len(top_entries)} candidates ...")

    # Build a lookup so generate_reasoning can access the full scores dict
    scores_by_cid: dict[str, dict[str, float]] = {
        cid: s for cid, s, _ in all_scores
    }

    output_rows: list[dict[str, Any]] = []
    for rank, (cid, raw_score, candidate) in enumerate(top_entries, start=1):
        reasoning = generate_reasoning(candidate, scores_by_cid[cid])
        output_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(raw_score, 6),
            "reasoning": reasoning,
        })

    # ------------------------------------------------------------------
    # Step 7 — validate before writing
    # ------------------------------------------------------------------
    print("  Validating output ...")
    errors = validate_submission(output_rows, top_k)
    if errors:
        print("\n[VALIDATION FAILED]", file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    wall_elapsed = time.perf_counter() - wall_start

    print(f"\n{'=' * 55}")
    print(f"  submission.csv written → {out_path}")
    print(f"  Candidates ranked: {total_candidates:,}")
    print(f"  Shortlisted:       {len(output_rows)}")
    print(f"  Top score:         {output_rows[0]['score']:.4f}")
    print(f"  Bottom score:      {output_rows[-1]['score']:.4f}")
    print(f"  Ranked {total_candidates:,} candidates in {wall_elapsed:.1f}s")
    print(f"{'=' * 55}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
