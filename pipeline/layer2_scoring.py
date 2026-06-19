"""
pipeline/layer2_scoring.py — Layer 2: Multi-Signal Fusion Scoring.

Takes the top candidates from Layer 1 and computes five independent
signals, then fuses them into a single weighted composite score (0–100).

Signals:
  1. Skills Match      (Tier 1, 40%) — semantic cosine similarity between JD
                                        skills and candidate skills via
                                        pre-computed all-MiniLM-L6-v2 embeddings.
  2. Career Trajectory (Tier 1, 20%) — growth arc derived from the title
                                        sequence (high_growth → 1.0,
                                        stagnant → 0.2).
  3. Recency           (Tier 2, 15%) — time-decay multiplier based on the
                                        end-date of the most recent role.
  4. Behavioral        (Tier 2, 15%) — profile completeness, GitHub presence,
                                        openness-to-move, experience recency.
  5. Seniority Align   (Tier 2, 10%) — bidirectional: underqualified AND
                                        overqualified both penalised.

All signals normalise to 0–1 before weighting.  Weights are taken from
JDIntent.weights (set by the JD parser) and fall back to module-level
defaults when a key is absent.

After fusion:
  — Hidden gem filter: skills ≥ 60 %, high-growth trajectory, score 60–79.
  — Low-match gate: warn when the top candidate scores < 40 / 100.

Usage:
    python -m pipeline.layer2_scoring
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import json
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logger import log_layer_complete, log_warning
from utils.schema import (
    CandidateFlags,
    CandidateNormalized,
    JDIntent,
    ScoredCandidate,
    SignalBreakdown,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "skills":     0.40,
    "trajectory": 0.20,
    "recency":    0.15,
    "behavioral": 0.15,
    "seniority":  0.10,
}

# Maps trajectory_label (from Layer 0) → raw score 0–1.
# Includes all labels produced by layer0_normalizer plus spec aliases.
TRAJECTORY_SCORES: dict[str | None, float] = {
    "high_growth":        1.0,
    "steady_climb":       0.7,   # layer0 label
    "steady":             0.7,   # spec alias
    "domain_expansion":   0.7,
    "consulting_pattern": 0.6,
    "lateral":            0.5,
    "startup_gap":        0.5,
    "declining":          0.1,   # layer0 label
    "stagnant":           0.2,
    "insufficient_data":  0.4,
    None:                 0.4,
}

# Expected experience-years range per JD seniority level.
SENIORITY_RANGES: dict[str, tuple[int, int]] = {
    "junior":  (0, 2),
    "mid":     (2, 5),
    "senior":  (4, 10),
    "lead":    (7, 15),
    "any":     (0, 99),
    "unknown": (0, 99),
}

# Tier thresholds (applied to fusion_score_100).
_TIER_THRESHOLDS = [
    (75.0, "strong_match"),
    (55.0, "good_match"),
    (40.0, "borderline"),
]


def _get_weight(jd_intent: JDIntent, key: str) -> float:
    """Return the JD-specific signal weight, falling back to the module default."""
    return jd_intent.weights.get(key, _DEFAULT_WEIGHTS[key])


# ===========================================================================
# SIGNAL 1 — SKILLS MATCH
# ===========================================================================

def precompute_skill_embeddings(
    jd_intent: JDIntent,
    candidates: list[CandidateNormalized],
    model: SentenceTransformer,
) -> dict[str, np.ndarray]:
    """
    Encode every unique skill string — from the JD and from all candidates —
    into a single batch and return a {skill_text: L2-normalised embedding} dict.

    Calling this once before the scoring loop avoids redundant encode() calls:
    each skill is encoded exactly once regardless of how many candidates share it.
    """
    all_skills: set[str] = set()
    all_skills.update(jd_intent.must_have)
    all_skills.update(jd_intent.nice_have)
    for c in candidates:
        all_skills.update(c.skills or [])

    skill_list = [s for s in all_skills if s]
    if not skill_list:
        return {}

    print(f"  → Precomputing embeddings for {len(skill_list)} unique skills...")
    embeddings: np.ndarray = model.encode(
        skill_list, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

    # L2-normalise so that cosine_sim(a, b) == np.dot(a, b)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings /= norms

    return {skill: emb for skill, emb in zip(skill_list, embeddings)}


def score_skills(
    candidate: CandidateNormalized,
    jd_intent: JDIntent,
    skill_embeddings: dict[str, np.ndarray],
    threshold: float = 0.75,
) -> tuple[SignalBreakdown, list[str], list[str]]:
    """
    Compute the skills-match signal using pre-computed embeddings.

    Threshold is adapted to JD specificity:
      — ≥ 6 must-have skills (niche role) → 0.65
      — ≤ 2 must-have skills (broad role)  → 0.80
      — otherwise                           → 0.75

    For each JD must-have skill the best cosine similarity against all
    candidate skills is found.  A match is counted when best_sim ≥ threshold.
    Nice-have skills follow the same logic but contribute only 30 % of the
    final raw score.

    Returns:
        (SignalBreakdown, skills_matched, skills_missing)
        skills_matched — candidate-side skill names that matched a JD skill
        skills_missing — JD must-have strings that were not matched
    """
    # Adaptive threshold
    n_must = len(jd_intent.must_have)
    if n_must >= 6:
        threshold = 0.65
    elif n_must <= 2:
        threshold = 0.80
    # else: keep the default 0.75

    cand_skills: list[str] = candidate.skills or []
    # Build parallel list of (name, embedding) only for skills present in the dict
    cand_pairs: list[tuple[str, np.ndarray]] = [
        (s, skill_embeddings[s]) for s in cand_skills if s in skill_embeddings
    ]

    matched: list[str] = []
    missing: list[str] = []

    def _best_match(jd_skill: str) -> tuple[float, str]:
        """Return (best_cosine_sim, best_matching_candidate_skill).

        If the candidate skill keyword appears literally inside the JD must_have
        phrase (e.g. 'TensorFlow' in '...PyTorch or TensorFlow...'), the cosine
        sim is boosted to 0.90 — a definitive lexical match that should always
        clear the threshold regardless of sentence-vs-keyword embedding distance.
        """
        jd_emb = skill_embeddings.get(jd_skill)
        if jd_emb is None or not cand_pairs:
            return 0.0, ""
        cand_names = [name for name, _ in cand_pairs]
        cand_embs = np.array([emb for _, emb in cand_pairs])  # (n, D)
        sims: np.ndarray = cand_embs @ jd_emb                  # (n,) cosine sims

        # Lexical boost: candidate skill keyword found verbatim in JD phrase
        jd_lower = jd_skill.lower()
        for idx, cname in enumerate(cand_names):
            if cname.lower() in jd_lower:
                sims[idx] = max(float(sims[idx]), 0.90)

        best_idx = int(sims.argmax())
        return float(sims[best_idx]), cand_names[best_idx]

    # Score must_have
    must_have_score = 0.0
    if jd_intent.must_have:
        matched_count = 0
        for jd_skill in jd_intent.must_have:
            best_sim, best_cand_skill = _best_match(jd_skill)
            if best_sim >= threshold:
                matched_count += 1
                if best_cand_skill:
                    matched.append(best_cand_skill)
            else:
                missing.append(jd_skill)
        must_have_score = matched_count / len(jd_intent.must_have)

    # Score nice_have
    nice_score = 0.0
    if jd_intent.nice_have:
        nice_matched = 0
        for jd_skill in jd_intent.nice_have:
            best_sim, _ = _best_match(jd_skill)
            if best_sim >= threshold:
                nice_matched += 1
        nice_score = nice_matched / len(jd_intent.nice_have)

    raw = round(0.7 * must_have_score + 0.3 * nice_score, 6)
    normalized = raw
    weighted = normalized * _get_weight(jd_intent, "skills")

    # Deduplicate while preserving order
    matched = list(dict.fromkeys(matched))

    return SignalBreakdown(raw, normalized, weighted), matched, missing


# ===========================================================================
# SIGNAL 2 — CAREER TRAJECTORY
# ===========================================================================

def score_trajectory(
    candidate: CandidateNormalized,
    jd_intent: JDIntent,
) -> SignalBreakdown:
    """
    Map the candidate's trajectory_label to a 0–1 raw score.

    Career changers are never penalised below 0.5 regardless of their
    trajectory label — a large gap or domain switch is not a red flag.
    """
    raw = TRAJECTORY_SCORES.get(candidate.trajectory_label, 0.4)

    # Career changer boost: never score below 0.5
    if candidate.is_career_changer:
        raw = max(raw, 0.5)

    normalized = raw
    weighted = normalized * _get_weight(jd_intent, "trajectory")
    return SignalBreakdown(raw, normalized, weighted)


# ===========================================================================
# SIGNAL 3 — RECENCY
# ===========================================================================

def _parse_ym(date_str: str) -> tuple[int, int] | None:
    """Parse a 'YYYY-MM' string into (year, month); returns None on failure."""
    try:
        parts = date_str.split("-")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError, AttributeError):
        return None


def score_recency(
    candidate: CandidateNormalized,
    jd_intent: JDIntent,
) -> SignalBreakdown:
    """
    Compute a time-decay multiplier based on the candidate's most recent role.

    Decay table:
      currently employed       → 1.00
      left < 1 year ago        → 0.95
      left 1–2 years ago       → 0.85
      left 2–3 years ago       → 0.75
      left 3–5 years ago       → 0.65
      left > 5 years ago       → 0.40
      no dated experience      → 0.50 (neutral, no penalty)
    """
    experience = candidate.experience or []
    now = datetime.now(timezone.utc)

    def _is_present(e) -> bool:
        return e.is_current or (
            e.end_date and e.end_date.strip().lower() in ("present", "now", "")
        )

    if not experience:
        raw = 0.5
    else:
        # Sort: present/active roles first, then most recent end_date desc
        def _sort_key(e):
            if _is_present(e):
                return (9999, 12)
            parsed = _parse_ym(e.end_date or "")
            return parsed if parsed else (0, 0)

        most_recent = max(experience, key=_sort_key)

        if _is_present(most_recent):
            years_since = 0.0
        else:
            parsed = _parse_ym(most_recent.end_date or "")
            if parsed is None:
                raw = 0.5
                normalized = raw
                weighted = normalized * _get_weight(jd_intent, "recency")
                return SignalBreakdown(raw, normalized, weighted)
            ey, em = parsed
            months_since = max(0, (now.year - ey) * 12 + (now.month - em))
            years_since = months_since / 12.0

        if years_since == 0.0:
            raw = 1.00
        elif years_since <= 1:
            raw = 0.95
        elif years_since <= 2:
            raw = 0.85
        elif years_since <= 3:
            raw = 0.75
        elif years_since <= 5:
            raw = 0.65
        else:
            raw = 0.40

    normalized = raw
    weighted = normalized * _get_weight(jd_intent, "recency")
    return SignalBreakdown(raw, normalized, weighted)


# ===========================================================================
# SIGNAL 4 — BEHAVIORAL
# ===========================================================================

def score_behavioral(
    candidate: CandidateNormalized,
    jd_intent: JDIntent,
) -> SignalBreakdown:
    """
    Composite behavioral signal from four equally-weighted sub-signals.

    Sub-signals (weights sum to 1.0):
      0.4 — profile completeness  (fields_present / total_fields)
      0.2 — GitHub presence       (1.0 if github_url exists, else 0.0)
      0.2 — openness to move      (inferred from raw_text + tenure signals)
      0.2 — experience recency    (actively employed vs gap)
    """
    now = datetime.now(timezone.utc)
    experience = candidate.experience or []

    # --- 1. Profile completeness ---
    n_present = len(candidate.fields_present or [])
    n_missing = len(candidate.fields_missing or [])
    total = n_present + n_missing
    completeness = n_present / total if total > 0 else 0.5

    # --- 2. GitHub presence ---
    github_presence = 1.0 if candidate.github_url else 0.0

    # --- 3. Openness to move ---
    text_lower = (candidate.raw_text or "").lower()
    openness_kws = ["looking for", "open to", "seeking", "opportunities"]

    if any(kw in text_lower for kw in openness_kws):
        openness = 1.0
    else:
        # Find the current (or most recent) role
        current_roles = [
            e for e in experience
            if e.is_current or (e.end_date and e.end_date.strip().lower() in ("present",))
        ]
        if current_roles:
            role = current_roles[0]
            parsed = _parse_ym(role.start_date or "")
            if parsed:
                sy, sm = parsed
                months_at = max(0, (now.year - sy) * 12 + (now.month - sm))
                if months_at >= 60:       # 5+ years at same company → entrenched
                    openness = 0.3
                elif months_at < 12:      # recently switched → active mover
                    openness = 0.6
                else:
                    openness = 0.5
            else:
                openness = 0.5
        else:
            openness = 0.5

    # --- 4. Experience recency (is the candidate currently active?) ---
    is_active = any(
        e.is_current or (e.end_date and e.end_date.strip().lower() in ("present",))
        for e in experience
    )

    if is_active:
        exp_recency = 1.0
    elif experience:
        # Find months since most recent role ended
        best_months: int | None = None
        for e in experience:
            parsed = _parse_ym(e.end_date or "")
            if parsed:
                ey, em = parsed
                months = max(0, (now.year - ey) * 12 + (now.month - em))
                if best_months is None or months < best_months:
                    best_months = months
        if best_months is None:
            exp_recency = 0.5
        elif best_months < 6:
            exp_recency = 0.8
        elif best_months < 12:
            exp_recency = 0.6
        else:
            exp_recency = 0.5
    else:
        exp_recency = 0.5

    raw = round(
        0.4 * completeness
        + 0.2 * github_presence
        + 0.2 * openness
        + 0.2 * exp_recency,
        6,
    )
    normalized = raw
    weighted = normalized * _get_weight(jd_intent, "behavioral")
    return SignalBreakdown(raw, normalized, weighted)


# ===========================================================================
# SIGNAL 5 — SENIORITY ALIGNMENT
# ===========================================================================

def score_seniority(
    candidate: CandidateNormalized,
    jd_intent: JDIntent,
) -> tuple[SignalBreakdown, str, bool]:
    """
    Bidirectional seniority alignment score.

    Both underqualified AND overqualified candidates are penalised relative
    to the JD seniority band:
      — overqualified: score decreases by 0.15 per year over the max
      — underqualified: score decreases by 0.20 per year under the min

    Fresher exception: a fresher candidate applying to a junior role scores
    0.8 regardless of years (they are a valid fit even at 0 years).

    Returns:
        (SignalBreakdown, seniority_label, is_overqualified)
    """
    exp_min, exp_max = SENIORITY_RANGES.get(
        jd_intent.seniority, SENIORITY_RANGES["unknown"]
    )
    candidate_years = candidate.total_years_exp or 0.0

    if exp_min <= candidate_years <= exp_max:
        raw = 1.0
        label = "aligned"
        is_overqualified = False
    elif candidate_years > exp_max:
        overshoot = candidate_years - exp_max
        raw = max(0.0, 1.0 - overshoot * 0.15)
        label = "overqualified"
        is_overqualified = True
    else:
        undershoot = exp_min - candidate_years
        raw = max(0.0, 1.0 - undershoot * 0.20)
        label = "underqualified"
        is_overqualified = False

    # Fresher exception: valid for junior roles
    if candidate.is_fresher and jd_intent.seniority == "junior":
        raw = 0.8
        label = "aligned"
        is_overqualified = False

    normalized = raw
    weighted = normalized * _get_weight(jd_intent, "seniority")
    return SignalBreakdown(raw, normalized, weighted), label, is_overqualified


# ===========================================================================
# FUSION
# ===========================================================================

def _determine_tier(score_100: float) -> str:
    """Map a 0–100 fusion score to a tier label."""
    for threshold, tier in _TIER_THRESHOLDS:
        if score_100 >= threshold:
            return tier
    return "weak_match"


def fuse_signals(
    candidate: CandidateNormalized,
    jd_intent: JDIntent,
    skill_embeddings: dict[str, np.ndarray],
) -> tuple[ScoredCandidate, float]:
    """
    Run all five signals and fuse them into a single ScoredCandidate.

    The fusion score is the sum of each signal's weighted contribution
    (already scaled by the signal weight).  Multiplied by 100 to yield
    a 0–100 score for human readability.

    Returns:
        (ScoredCandidate with rank=0, fusion_score_100)
        The caller is responsible for sorting and assigning final ranks.
    """
    # Score all signals
    skills_bd, skills_matched, skills_missing = score_skills(
        candidate, jd_intent, skill_embeddings
    )
    trajectory_bd = score_trajectory(candidate, jd_intent)
    recency_bd = score_recency(candidate, jd_intent)
    behavioral_bd = score_behavioral(candidate, jd_intent)
    seniority_bd, seniority_label, is_overqualified = score_seniority(
        candidate, jd_intent
    )

    # Fuse
    fusion_score = (
        skills_bd.weighted
        + trajectory_bd.weighted
        + recency_bd.weighted
        + behavioral_bd.weighted
        + seniority_bd.weighted
    )
    fusion_score_100 = round(fusion_score * 100, 2)

    breakdown: dict[str, SignalBreakdown] = {
        "skills":     skills_bd,
        "trajectory": trajectory_bd,
        "recency":    recency_bd,
        "behavioral": behavioral_bd,
        "seniority":  seniority_bd,
    }

    flags = CandidateFlags(
        is_fresher=candidate.is_fresher,
        is_overqualified=is_overqualified,
        is_duplicate=False,
        incomplete_profile=(candidate.confidence_level == "LOW"),
        score_conflict=False,
        low_match=False,
        llm_rerank=False,           # set True in Layer 3 after re-ranking
        github_verified=False,      # set in layer2_github
        reasoning_verified=True,
        is_career_changer=candidate.is_career_changer,
        is_hidden_gem=False,
    )

    tier = _determine_tier(fusion_score_100)

    sc = ScoredCandidate(
        rank=0,
        candidate_id=candidate.candidate_id,
        final_score=fusion_score_100,
        signal_fusion_score=fusion_score_100,
        llm_rerank_score=None,
        signal_breakdown=breakdown,
        confidence_level=candidate.confidence_level,
        flags=flags,
        tier=tier,
        reasoning=None,
        skills_matched=skills_matched,
        skills_missing=skills_missing,
        experience_years=candidate.total_years_exp,
        trajectory_label=candidate.trajectory_label,
        seniority_label=seniority_label,
        location_match=None,
        github_profile=None,
    )
    return sc, fusion_score_100


# ===========================================================================
# LOW MATCH GATE
# ===========================================================================

def check_low_match(scored: list[ScoredCandidate]) -> bool:
    """
    Return True (fire low-match warning) when the top candidate's fusion
    score is below 40 / 100, indicating the JD and candidate pool are a
    poor overall fit.
    """
    if not scored:
        return True
    return scored[0].signal_fusion_score < 40.0


# ===========================================================================
# HIDDEN GEM DETECTION
# ===========================================================================

def detect_hidden_gems(
    scored: list[ScoredCandidate],
    candidates_map: dict[str, CandidateNormalized],
    main_shortlist_ids: set[str],
) -> list[ScoredCandidate]:
    """
    Identify candidates who are undervalued by the fusion score but show
    strong trajectory and skills signals, and were not captured in the main
    shortlist top 10.

    Criteria (all must hold):
      — skills signal normalised score ≥ 0.35
      — trajectory_label in {'high_growth', 'domain_expansion', 'steady_climb'}
      — 45.0 ≤ fusion_score_100 ≤ 74.0
      — candidate_id NOT in main_shortlist_ids (i.e. not already in top 10)

    Qualifying candidates are sorted by trajectory score desc, then skills
    score desc, and capped at 5.  Their tier is mutated to 'hidden_gem' and
    is_hidden_gem flag set.
    """
    HIGH_TRAJECTORY = {"high_growth", "domain_expansion", "steady_climb"}
    gems: list[ScoredCandidate] = []

    for sc in scored:
        skills_norm = sc.signal_breakdown["skills"].normalized
        traj = sc.trajectory_label or ""
        score = sc.signal_fusion_score

        if (
            skills_norm >= 0.35
            and traj in HIGH_TRAJECTORY
            and 45.0 <= score <= 74.0
            and sc.candidate_id not in main_shortlist_ids
        ):
            gems.append(sc)

    gems.sort(
        key=lambda c: (
            TRAJECTORY_SCORES.get(c.trajectory_label, 0.4),
            c.signal_breakdown["skills"].normalized,
        ),
        reverse=True,
    )
    gems = gems[:5]

    for gem in gems:
        gem.tier = "hidden_gem"
        gem.flags.is_hidden_gem = True

    return gems


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def run_layer2(
    jd_intent: JDIntent,
    candidates: list[CandidateNormalized],
    model: SentenceTransformer,
    top_k: int = 50,
) -> tuple[list[ScoredCandidate], list[ScoredCandidate], bool]:
    """
    Layer 2 top-level orchestrator — multi-signal fusion pipeline.

    Accepts the SentenceTransformer model as a parameter so it can be
    shared from Layer 1 without reloading (~90 MB checkpoint).

    Steps:
      1. Precompute skill embeddings (one batch encode).
      2. Score each candidate across all 5 signals (tqdm progress).
      3. Sort by fusion_score_100 descending.
      4. Assign 1-indexed ranks.
      5. Check low-match gate; log warning if triggered.
      6. Detect hidden gems (mutates tier + flag on qualifying objects).
      7. Slice sorted list to top_k for shortlist.
      8. Log layer completion.

    Returns:
        (shortlist, hidden_gems, low_match_warning)
        shortlist     — top_k ScoredCandidate objects, ranked 1..top_k
        hidden_gems   — up to 5 hidden-gem candidates (never in shortlist top 10)
        low_match_warning — True when top score < 40 / 100
    """
    start = time.time()

    if not candidates:
        log_warning("Layer 2: received 0 candidates — returning empty lists.")
        return [], [], True

    # Step 1 — precompute all skill embeddings in one shot
    skill_embeddings = precompute_skill_embeddings(jd_intent, candidates, model)

    # Step 2 — score each candidate
    scored_pairs: list[tuple[ScoredCandidate, float]] = []
    for c in tqdm(candidates, desc="  Scoring candidates", unit="cand"):
        sc, score = fuse_signals(c, jd_intent, skill_embeddings)
        scored_pairs.append((sc, score))

    # Step 3 — sort descending by fusion score
    scored_pairs.sort(key=lambda p: p[1], reverse=True)
    scored: list[ScoredCandidate] = [sc for sc, _ in scored_pairs]

    # Step 4 — assign 1-indexed ranks
    for rank, sc in enumerate(scored, start=1):
        sc.rank = rank

    # Step 5 — low-match gate
    low_match_warning = check_low_match(scored)
    if low_match_warning:
        log_warning(
            f"Low match warning: top candidate score = "
            f"{scored[0].signal_fusion_score:.1f} / 100 (threshold: 40)"
        )

    # Step 6 — slice to top_k for the main shortlist
    shortlist = scored[:top_k]

    # Step 7 — hidden gems: scan the full scored list for candidates outside
    # the top 10 who show strong signals the recruiter might have missed.
    candidates_map: dict[str, CandidateNormalized] = {
        c.candidate_id: c for c in candidates
    }
    main_shortlist_ids = {s.candidate_id for s in shortlist[:10]}
    hidden_gems = detect_hidden_gems(scored, candidates_map, main_shortlist_ids)

    elapsed = time.time() - start
    log_layer_complete(layer=2, candidates_remaining=len(shortlist), elapsed_seconds=elapsed)

    return shortlist, hidden_gems, low_match_warning


# ===========================================================================
# QUICK TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer as _ST

    from pipeline.layer0_normalizer import run_layer0
    from pipeline.layer1_retrieval import run_layer1

    DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

    # --- Load JD ---
    jd_raw = json.loads((DATA_DIR / "jd_senior_ml.txt").read_text(encoding="utf-8"))
    jd_obj = jd_raw.get("job_description", jd_raw)
    jd_parts: list[str] = []
    for key in (
        "role_overview", "responsibilities", "must_have_requirements",
        "nice_to_have", "preferred_qualifications", "technical_requirements",
        "compensation_and_benefits", "about_the_role",
    ):
        val = jd_obj.get(key)
        if isinstance(val, str):
            jd_parts.append(val)
        elif isinstance(val, list):
            jd_parts.extend(str(v) for v in val)
    jd_text = "\n".join(jd_parts) or json.dumps(jd_obj)

    # --- Load candidates ---
    all_candidates: list[dict] = json.loads(
        (DATA_DIR / "candidates.json").read_text(encoding="utf-8")
    )
    type_by_id: dict[str, str] = {
        c["candidate_id"]: c.get("type", "") for c in all_candidates
    }

    print("\n" + "=" * 70)
    print("  Layer 2 Quick Test — jd_senior_ml.txt + 30 candidates")
    print("=" * 70 + "\n")

    # --- Layer 0 ---
    jd_intent, normalized = run_layer0(jd_text, "JD_SENIOR_ML", all_candidates)
    print(f"\n  Layer 0: {len(normalized)} candidates normalised.\n")

    # --- Layer 1 ---
    retrieved, model = run_layer1(jd_intent, normalized, top_k=50)
    print(f"\n  Layer 1: {len(retrieved)} candidates retrieved.\n")

    # Reuse model loaded by Layer 1 (avoids a second ~90 MB checkpoint load)
    if model is None:
        print("  Layer 1 returned no model — loading separately...")
        model = _ST("all-MiniLM-L6-v2")

    # --- Layer 2 ---
    shortlist, hidden_gems, low_match_warning = run_layer2(
        jd_intent, retrieved, model, top_k=50
    )
    print(f"\n  Layer 2: {len(shortlist)} candidates scored.\n")

    # --- Results table (top 15) ---
    hdr = (
        f"  {'Rank':>4} | {'ID':<5} | {'Type':<22} | "
        f"{'Fusion':>6} | {'Skills':>6} | {'Traj':>5} | "
        f"{'Recency':>7} | {'Seniority':>9} | Tier"
    )
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for sc in shortlist[:15]:
        bd = sc.signal_breakdown
        ctype = type_by_id.get(sc.candidate_id, "")
        print(
            f"  {sc.rank:>4} | {sc.candidate_id:<5} | {ctype:<22} | "
            f"{sc.signal_fusion_score:>6.1f} | "
            f"{bd['skills'].normalized:>6.2f} | "
            f"{bd['trajectory'].normalized:>5.2f} | "
            f"{bd['recency'].normalized:>7.2f} | "
            f"{bd['seniority'].normalized:>9.2f} | "
            f"{sc.tier}"
        )

    # --- Hidden gems ---
    print(f"\n  Hidden gems ({len(hidden_gems)}):")
    if hidden_gems:
        for gem in hidden_gems:
            bd = gem.signal_breakdown
            ctype = type_by_id.get(gem.candidate_id, "")
            print(
                f"    {gem.candidate_id} [{ctype}] "
                f"fusion={gem.signal_fusion_score:.1f} "
                f"skills={bd['skills'].normalized:.2f} "
                f"traj={gem.trajectory_label}"
            )
    else:
        print("    (none)")

    print(f"\n  low_match_warning = {low_match_warning}")

    # --- Assertions ---
    top1 = shortlist[0]
    assert top1.tier in ("strong_match", "good_match"), (
        f"Expected top candidate to be strong_match/good_match, got {top1.tier!r} "
        f"(score={top1.signal_fusion_score:.1f})"
    )

    top5_types = [type_by_id.get(sc.candidate_id, "") for sc in shortlist[:5]]
    assert "poor_match" not in top5_types, (
        f"Found poor_match in top 5: {top5_types}"
    )

    assert not low_match_warning, (
        f"Expected low_match_warning=False, got True "
        f"(top score={shortlist[0].signal_fusion_score:.1f})"
    )

    overq_ids = {
        c["candidate_id"] for c in all_candidates if c.get("type") == "overqualified"
    }
    for sc in shortlist:
        if sc.candidate_id in overq_ids:
            assert sc.flags.is_overqualified, (
                f"Expected {sc.candidate_id} to be flagged is_overqualified=True"
            )

    print("\n  ✓ All Layer 2 assertions passed.")
