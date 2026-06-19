"""
Central schema definitions for the HireSynth pipeline.

Every dataclass and enum used across all pipeline layers is defined here.
All other modules import their types exclusively from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExitStatus(str, Enum):
    """Pipeline-level exit codes describing how the run concluded."""

    EXIT_SUCCESS = "EXIT_SUCCESS"
    EXIT_LOW_MATCH = "EXIT_LOW_MATCH"
    EXIT_NO_CANDIDATES = "EXIT_NO_CANDIDATES"
    EXIT_JD_PARSE_FAIL = "EXIT_JD_PARSE_FAIL"
    EXIT_API_DOWN = "EXIT_API_DOWN"


# ---------------------------------------------------------------------------
# Layer 0 — JD Parser output
# ---------------------------------------------------------------------------

@dataclass
class JDIntent:
    """Structured intent object produced by the JD parser (Layer 0, left stream)."""

    job_id: str
    job_title: str
    must_have: List[str]
    nice_have: List[str]
    seniority: str                          # e.g. "senior", "mid", "junior"
    domain: str                             # e.g. "ml", "mlops", "data_science"
    culture_signals: List[str]
    implicit_needs: List[str]
    retrieval_mode: str                     # "technical" | "semantic"
    weights: Dict[str, float]              # signal name → weight (0–1, sum ≈ 1)
    quality_score: float                   # 0–100
    quality_level: str                     # "HIGH" | "MEDIUM" | "LOW"
    is_multi_role: bool = False
    primary_track: Optional[Dict] = None
    secondary_track: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Layer 0 — Resume Normalizer types
# ---------------------------------------------------------------------------

@dataclass
class ExperienceEntry:
    """A single role in a candidate's work history."""

    title: str
    company: str
    start_date: Optional[str] = None       # ISO-8601 or free-text
    end_date: Optional[str] = None         # None / "Present" if is_current
    duration_years: Optional[float] = None
    summary: Optional[str] = None
    is_current: bool = False


@dataclass
class CandidateRaw:
    """Raw input record before any parsing — minimal identity + text blob."""

    candidate_id: str
    raw_text: str
    source_file: Optional[str] = None


@dataclass
class CandidateNormalized:
    """
    Normalizer output for a single candidate (Layer 0, right stream).

    All fields are nullable; only reward what is present — never penalize
    what is missing.
    """

    candidate_id: str
    raw_text: str
    name: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experience: List[ExperienceEntry] = field(default_factory=list)
    education: Optional[List[Dict]] = None
    location: Optional[str] = None
    total_years_exp: Optional[float] = None
    trajectory_label: Optional[str] = None     # "high_growth" | "stable" | "declining" | …
    github_url: Optional[str] = None
    confidence_level: str = "LOW"               # "HIGH" | "MEDIUM" | "LOW"
    language_detected: Optional[str] = None     # ISO 639-1 language code
    is_fresher: bool = False
    is_career_changer: bool = False
    fields_present: List[str] = field(default_factory=list)
    fields_inferred: List[str] = field(default_factory=list)
    fields_missing: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GitHub Enrichment (between Layer 2 and Layer 3)
# ---------------------------------------------------------------------------

@dataclass
class GitHubProfile:
    """GitHub API enrichment data for a single candidate."""

    username: str
    public_repos: int
    total_stars: int
    top_languages: List[str]
    last_push_date: Optional[str] = None   # ISO-8601 date string
    account_age_years: Optional[float] = None
    followers: int = 0
    github_score: float = 0.0              # 0–1, boost-only contribution
    verified: bool = False


# ---------------------------------------------------------------------------
# Layer 2 — Signal Fusion scoring types
# ---------------------------------------------------------------------------

@dataclass
class SignalBreakdown:
    """Score detail for one signal: raw value, normalized value, weighted contribution."""

    raw: float          # signal output before normalization (0–1)
    normalized: float   # after normalization (0–1)
    weighted: float     # normalized × signal weight, in final-score points


# ---------------------------------------------------------------------------
# Layer 3 — LLM Re-ranker types
# ---------------------------------------------------------------------------

@dataclass
class ReasoningCard:
    """
    Structured reasoning produced by the LLM re-ranker (Layer 3).

    Three mandatory sentences plus a 1–10 score.
    """

    strength: str                        # top strength in one sentence
    gap: str                             # key gap or risk in one sentence
    recommendation: str                  # recruiter recommendation in one sentence
    llm_score: int                       # 1–10
    verified: bool = True                # False when fusion/LLM conflict detected
    conflict_note: Optional[str] = None  # set when |llm_score - fusion_score| > 15


# ---------------------------------------------------------------------------
# Candidate output flags
# ---------------------------------------------------------------------------

@dataclass
class CandidateFlags:
    """All boolean flags that can be set on a scored candidate."""

    is_fresher: bool = False
    is_overqualified: bool = False
    is_duplicate: bool = False
    incomplete_profile: bool = False
    score_conflict: bool = False
    low_match: bool = False
    llm_rerank: bool = True
    github_verified: bool = False
    reasoning_verified: bool = True
    is_career_changer: bool = False
    is_hidden_gem: bool = False


# ---------------------------------------------------------------------------
# Full scored-candidate record
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """
    Complete per-candidate output record written to the shortlist.

    Aggregates scores from every pipeline layer plus all derived metadata.
    """

    rank: int
    candidate_id: str
    final_score: float                                  # 0–100
    signal_fusion_score: float                          # 0–100, pre-LLM
    llm_rerank_score: Optional[float]                  # 0–100, None if API down
    signal_breakdown: Dict[str, SignalBreakdown]       # signal name → breakdown
    confidence_level: str                               # "HIGH" | "MEDIUM" | "LOW"
    flags: CandidateFlags
    tier: str                                           # "strong_match" | "good_match" | "hidden_gem" | …
    reasoning: Optional[ReasoningCard]
    skills_matched: List[str] = field(default_factory=list)
    skills_missing: List[str] = field(default_factory=list)
    experience_years: Optional[float] = None
    trajectory_label: Optional[str] = None
    seniority_label: Optional[str] = None
    location_match: Optional[str] = None               # "exact" | "country" | "remote" | "mismatch"
    github_profile: Optional[GitHubProfile] = None


# ---------------------------------------------------------------------------
# Pipeline run metadata
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetadata:
    """Run-level metadata emitted once per pipeline execution."""

    job_id: str
    job_title: str
    jd_quality_score: float
    jd_quality_level: str
    total_candidates_processed: int
    total_candidates_shortlisted: int
    low_match_warning: bool
    pipeline_version: str
    generated_at: str                           # ISO-8601 datetime string
    run_hash: str                               # md5(jd_text + sorted(candidate_ids))[:8]
    exit_status: ExitStatus
    candidates_at_each_stage: Dict[str, int]   # stage name → candidate count
    signal_weights_used: Dict[str, float]      # signal name → weight applied
    skill_threshold_used: float
    embedding_model: str
    hidden_gems_note: Optional[str] = None     # populated when hidden_gems list is empty


# ---------------------------------------------------------------------------
# Top-level pipeline output
# ---------------------------------------------------------------------------

@dataclass
class PipelineOutput:
    """
    Complete output file structure — written as JSON after every run.

    Maps directly to the locked output schema defined in CLAUDE.md.
    """

    metadata: PipelineMetadata
    shortlist: List[ScoredCandidate]
    hidden_gems: List[ScoredCandidate]
    evaluation: Dict                # ndcg_score, confidence_interval, eval_set_size, note
    audit_log: Dict                 # run_id, embedding_model, jd_quality_score, …
    bias_note: str
