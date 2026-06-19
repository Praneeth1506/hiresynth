"""
pipeline/layer3_reranker.py — Layer 3: LLM Re-ranking.

Takes the top 50 scored candidates from Layer 2 (post-GitHub enrichment)
and re-ranks them using GPT-4o-mini reasoning.  Each candidate receives a
structured three-sentence reasoning card (strength / gap / recommendation)
plus a 1–10 LLM score that is blended with the fusion score.

Design principles:
  — Candidate anonymisation: only Candidate_ID is sent, never name.
  — temperature=0 on all LLM calls for deterministic output.
  — asyncio.Semaphore(10) caps concurrent API calls.
  — Candidate summaries are capped at 500 tokens (≈ 2 000 chars).
  — Ranking always uses fusion score; LLM score is informational only.
  — Conflict flag raised when |llm_score − fusion_score| > 15 pts.
  — API fallback: on failure → use L2 score, flag llm_rerank=False.

Usage:
    python -m pipeline.layer3_reranker
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Optional

import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

from utils.logger import log_layer_complete, log_warning
from utils.schema import (
    CandidateNormalized,
    JDIntent,
    ReasoningCard,
    ScoredCandidate,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MODEL = "gpt-4o-mini"
_MAX_RESPONSE_TOKENS = 150
_SEMAPHORE_LIMIT = 10


# ===========================================================================
# PART 1 — CANDIDATE SUMMARY BUILDER
# ===========================================================================

def _format_education(education: Optional[list | dict]) -> str:
    """
    Render an education field (list[dict] or single dict) as a one-line string.

    Returns 'Not specified' when the field is absent or empty.
    """
    if not education:
        return "Not specified"
    entry = education[0] if isinstance(education, list) else education
    if not isinstance(entry, dict):
        return str(entry)
    degree = entry.get("degree", "")
    institution = entry.get("institution", "")
    year = entry.get("graduation_year", "")
    parts: list[str] = []
    if degree:
        parts.append(degree)
    if institution:
        parts.append(f"from {institution}")
    if year:
        parts.append(f"({year})")
    return " ".join(parts) if parts else "Not specified"


def build_candidate_summary(
    scored: ScoredCandidate,
    candidate: CandidateNormalized,
    max_tokens: int = 500,
) -> str:
    """
    Serialise a scored candidate into a structured text block for the LLM.

    Uses Candidate_ID, never name (bias prevention).  Includes the last 3
    experience entries, top 10 skills, education, trajectory, signal scores,
    and GitHub score when available.  Total length is capped at
    max_tokens * 4 characters (approximate token budget at 4 chars/token).

    Returns:
        Multi-line string ready to embed in the user prompt.
    """
    bd = scored.signal_breakdown

    # --- Skills (top 10) ---
    skills_str = ", ".join((candidate.skills or [])[:10]) or "Not specified"

    # --- Experience (last 3 roles = first 3 in list, most recent first) ---
    exp_lines: list[str] = []
    for e in (candidate.experience or [])[:3]:
        end = e.end_date or "present"
        exp_lines.append(f"  {e.title} @ {e.company} ({e.start_date}→{end})")
    experience_str = "\n".join(exp_lines) if exp_lines else "  Not specified"

    # --- Education ---
    education_str = _format_education(candidate.education)

    # --- Signal score lines ---
    skills_pct = f"{bd['skills'].normalized:.0%}" if "skills" in bd else "N/A"
    traj_pct = f"{bd['trajectory'].normalized:.0%}" if "trajectory" in bd else "N/A"
    recency_pct = f"{bd['recency'].normalized:.0%}" if "recency" in bd else "N/A"

    # --- Optional GitHub score ---
    github_line = ""
    if scored.github_profile and scored.github_profile.verified:
        github_line = (
            f"\nGITHUB SCORE: {scored.github_profile.github_score:.2f}"
            f"  (repos={scored.github_profile.public_repos},"
            f" stars={scored.github_profile.total_stars},"
            f" languages={', '.join(scored.github_profile.top_languages[:3])})"
        )

    summary = (
        f"CANDIDATE: {scored.candidate_id}\n"
        f"SKILLS: {skills_str}\n"
        f"EXPERIENCE:\n{experience_str}\n"
        f"TOTAL EXPERIENCE: {candidate.total_years_exp or 0:.1f} years\n"
        f"EDUCATION: {education_str}\n"
        f"LOCATION: {candidate.location or 'Not specified'}\n"
        f"TRAJECTORY: {candidate.trajectory_label or 'unknown'}"
        f"{github_line}\n"
        f"SIGNAL SCORES:\n"
        f"  Skills match:  {skills_pct}\n"
        f"  Trajectory:    {traj_pct}\n"
        f"  Recency:       {recency_pct}\n"
        f"  Seniority:     {scored.seniority_label or 'unknown'}\n"
        f"FUSION SCORE: {scored.signal_fusion_score:.1f}/100"
    )

    # Hard truncate to max_tokens * 4 characters
    char_limit = max_tokens * 4
    if len(summary) > char_limit:
        summary = summary[:char_limit]

    return summary


# ===========================================================================
# PART 2 — REASONING PROMPT
# ===========================================================================

def build_rerank_prompt(
    candidate_summary: str,
    jd_intent: JDIntent,
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) pair sent to GPT-4o-mini.

    The system prompt establishes the recruiter persona and strict format
    adherence.  The user prompt embeds the role requirements and candidate
    profile, and enforces the four-line output contract.

    Returns:
        (system_prompt, user_prompt) tuple of plain strings.
    """
    system_prompt = (
        "You are an expert technical recruiter.\n"
        "Evaluate candidates against job requirements.\n"
        "Be precise, honest, and concise.\n"
        "Always respond in exactly the format requested.\n"
        "Never add extra commentary outside the format."
    )

    must_have_str = ", ".join(jd_intent.must_have)
    nice_have_str = ", ".join(jd_intent.nice_have)

    user_prompt = (
        "Evaluate this candidate for the following role.\n\n"
        "ROLE REQUIREMENTS:\n"
        f"Must have: {must_have_str}\n"
        f"Nice to have: {nice_have_str}\n"
        f"Seniority: {jd_intent.seniority}\n"
        f"Domain: {jd_intent.domain}\n\n"
        "CANDIDATE PROFILE:\n"
        f"{candidate_summary}\n\n"
        "Respond in EXACTLY this format, nothing else:\n"
        "STRENGTH: [one sentence — strongest qualification]\n"
        "GAP: [one sentence — key missing skill or risk]\n"
        "RECOMMENDATION: [one sentence — hire/consider/pass and why]\n"
        "SCORE: [integer 1-10]\n\n"
        "Rules:\n"
        "- Each line starts with the exact label shown\n"
        "- SCORE must be a single integer, nothing else\n"
        "- Do not add any other text or explanation"
    )

    return system_prompt, user_prompt


# ===========================================================================
# PART 3 — RESPONSE PARSER
# ===========================================================================

def parse_reasoning_response(
    response_text: str,
    candidate_id: str,
    fusion_score: float,
) -> tuple[ReasoningCard, float, bool]:
    """
    Parse the four-line LLM output into a ReasoningCard plus score metadata.

    Expected format:
        STRENGTH: <sentence>
        GAP: <sentence>
        RECOMMENDATION: <sentence>
        SCORE: <int 1-10>

    Fallback when any field is missing or SCORE is unparseable:
        strength = gap = "Unable to parse"
        recommendation = "Manual review recommended"
        llm_score (1-10) = int(fusion_score / 10)
        llm_score_100 = that value * 10

    Conflict detection:
        conflict     = |llm_score_100 − fusion_score| > 15
        verified     = False  if (llm_strong AND fusion < 50)
                              OR (llm_weak  AND fusion > 80)
                       True   otherwise

    Returns:
        (ReasoningCard, llm_score_100, verified)
    """
    _label_re = {
        "STRENGTH":       re.compile(r"^STRENGTH:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "GAP":            re.compile(r"^GAP:\s*(.+)$",      re.IGNORECASE | re.MULTILINE),
        "RECOMMENDATION": re.compile(r"^RECOMMENDATION:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
        "SCORE":          re.compile(r"^SCORE:\s*(\d+)",    re.IGNORECASE | re.MULTILINE),
    }

    def _extract(key: str) -> str | None:
        m = _label_re[key].search(response_text or "")
        return m.group(1).strip() if m else None

    strength       = _extract("STRENGTH")
    gap            = _extract("GAP")
    recommendation = _extract("RECOMMENDATION")
    score_raw      = _extract("SCORE")

    parse_ok = all([strength, gap, recommendation, score_raw])

    if parse_ok:
        try:
            score_1_10: int = max(1, min(10, int(score_raw)))  # clamp to [1,10]
        except (ValueError, TypeError):
            parse_ok = False

    if not parse_ok:
        if response_text and response_text.strip():
            log_warning(
                f"Layer 3: failed to parse LLM response for {candidate_id}. "
                f"Using fusion-score fallback."
            )
        strength = "Unable to parse"
        gap = "Unable to parse"
        recommendation = "Manual review recommended"
        score_1_10 = max(1, min(10, int(fusion_score / 10)))

    llm_score_100 = float(score_1_10 * 10)

    # Conflict detection
    conflict = abs(llm_score_100 - fusion_score) > 15
    llm_strong = score_1_10 >= 7
    llm_weak   = score_1_10 <= 3

    if (llm_strong and fusion_score < 50) or (llm_weak and fusion_score > 80):
        verified = False
    else:
        verified = True

    card = ReasoningCard(
        strength=strength,
        gap=gap,
        recommendation=recommendation,
        llm_score=score_1_10,
        verified=verified,
    )

    return card, llm_score_100, verified


# ===========================================================================
# PART 4 — ASYNC LLM CALLER
# ===========================================================================

async def call_llm_async(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Call GPT-4o-mini with a structured prompt and return the raw text response.

    Concurrency is limited by the shared asyncio.Semaphore (max 10 in-flight).
    temperature=0 ensures deterministic output across re-runs.

    Returns:
        Raw response content string on success; empty string on any exception
        (the empty string triggers safe-default parsing in parse_reasoning_response).
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=_MODEL,
                temperature=0,
                seed=42,
                max_tokens=_MAX_RESPONSE_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            log_warning(f"Layer 3: LLM call failed — {exc}")
            return ""


async def rerank_all_async(
    candidates_with_summaries: list[tuple[ScoredCandidate, str]],
    jd_intent: JDIntent,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> list[tuple[ScoredCandidate, ReasoningCard, float, bool]]:
    """
    Fire all LLM calls concurrently using asyncio.gather, then parse responses.

    Each candidate's prompt is built, launched as a coroutine, and gathered in
    a single await.  Parsing happens sequentially after all responses arrive.

    Returns:
        List of (ScoredCandidate, ReasoningCard, llm_score_100, api_success)
        where api_success=False when the LLM returned an empty string (error /
        timeout) and triggers the llm_rerank=False fallback path.
    """
    # Build prompts and launch coroutines
    coroutines = []
    for scored, summary in candidates_with_summaries:
        sys_p, usr_p = build_rerank_prompt(summary, jd_intent)
        coroutines.append(call_llm_async(client, sys_p, usr_p, semaphore))

    responses: list[str] = await asyncio.gather(*coroutines)

    # Parse responses
    results: list[tuple[ScoredCandidate, ReasoningCard, float, bool]] = []
    for (scored, _summary), response in zip(candidates_with_summaries, responses):
        api_success = bool(response.strip())
        card, llm_score_100, _verified = parse_reasoning_response(
            response, scored.candidate_id, scored.signal_fusion_score
        )
        results.append((scored, card, llm_score_100, api_success))

    return results


# ===========================================================================
# PART 5 — SCORE MERGER
# ===========================================================================

def merge_scores(
    scored: ScoredCandidate,
    reasoning: ReasoningCard,
    llm_score_100: float,
    api_success: bool = True,
) -> ScoredCandidate:
    """
    Attach LLM reasoning to the ScoredCandidate without touching final_score.

    final_score is set by Layer 2 (signal fusion) and optionally boosted by
    the GitHub enrichment step.  merge_scores() must never overwrite it —
    doing so would either introduce LLM non-determinism or silently strip the
    GitHub boost.

    What this function does:
      — Stores llm_score_100 in llm_rerank_score (display only).
      — Attaches the reasoning card.
      — Sets llm_rerank / score_conflict / reasoning_verified flags.
      — Appends a note to recommendation when conflict is detected (> 15 pts).

    What this function does NOT do:
      — Modify final_score in any path (LLM has zero influence on rank order).

    Mutates the ScoredCandidate in-place and returns it.
    """
    conflict = abs(llm_score_100 - scored.signal_fusion_score) > 15

    if not api_success:
        scored.flags.llm_rerank = False
    else:
        scored.flags.llm_rerank = True
        if conflict:
            reasoning.recommendation = (
                reasoning.recommendation
                + " [Note: LLM assessment diverged significantly from signal"
                " fusion score — manual review recommended.]"
            )
            log_warning(
                f"Layer 3 conflict — {scored.candidate_id}: "
                f"llm={llm_score_100:.0f} vs fusion={scored.signal_fusion_score:.1f} "
                f"(delta={abs(llm_score_100 - scored.signal_fusion_score):.1f}). "
                f"Fusion score used for ranking."
            )

    # final_score is intentionally left unchanged — it already equals
    # signal_fusion_score + github_boost (applied before Layer 3).
    scored.llm_rerank_score = llm_score_100
    scored.reasoning = reasoning
    scored.flags.score_conflict = conflict
    scored.flags.reasoning_verified = reasoning.verified

    return scored


# ===========================================================================
# PART 6 — LLM RESPONSE CACHE
# ===========================================================================

_CACHE_PATH = Path(__file__).parent.parent / "data" / "llm_cache.json"
_CACHE_VERSION = "1.0"


def _compute_cache_key(jd_text: str, candidate_ids: list[str]) -> str:
    """Compute a stable 8-char md5 key from JD text and sorted candidate IDs."""
    candidates_repr = "".join(sorted(candidate_ids))
    raw = (jd_text + candidates_repr).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:8]


def _load_llm_cache(cache_key: str) -> Optional[dict]:
    """
    Load cached LLM responses if the cache file exists and the key matches.

    Returns the responses dict on a hit, None on any miss or parse error.
    """
    if not _CACHE_PATH.exists():
        return None
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if (
            data.get("cache_version") == _CACHE_VERSION
            and data.get("jd_hash") == cache_key
        ):
            return data.get("responses", {})
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def _save_llm_cache(cache_key: str, responses: dict) -> None:
    """Persist LLM responses to the cache file, creating directories as needed."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "cache_version": _CACHE_VERSION,
        "jd_hash": cache_key,
        "responses": responses,
    }
    _CACHE_PATH.write_text(
        json.dumps(cache_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ===========================================================================
# PART 7 — ORCHESTRATOR
# ===========================================================================

def run_layer3(
    jd_intent: JDIntent,
    scored_candidates: list[ScoredCandidate],
    candidates_map: dict[str, CandidateNormalized],
    top_k: int = 20,
    use_cache: bool = True,
    jd_text: str = "",
) -> list[ScoredCandidate]:
    """
    Layer 3 top-level orchestrator — LLM re-ranking pipeline.

    Steps:
      1. Take up to 50 candidates (or all if fewer).
      2. Build a capped text summary for each candidate.
      3. Fire all LLM calls concurrently via asyncio (Semaphore=10).
      4. Merge LLM scores with fusion scores per candidate.
      5. Re-sort by final_score descending and re-assign 1-indexed ranks.
      6. Slice to top_k for the final shortlist.
      7. Log layer completion.

    Returns:
        List of up to top_k ScoredCandidate objects, re-ranked by final_score.
    """
    start = time.time()

    if not scored_candidates:
        log_warning("Layer 3: received 0 candidates — returning empty list.")
        return []

    # Step 1 — cap at 50
    top_50 = scored_candidates[:50]

    # Step 2 — build summaries
    candidates_with_summaries: list[tuple[ScoredCandidate, str]] = []
    for sc in top_50:
        norm = candidates_map.get(sc.candidate_id)
        if norm is None:
            # Provide minimal summary when mapping is missing
            summary = (
                f"CANDIDATE: {sc.candidate_id}\n"
                f"FUSION SCORE: {sc.signal_fusion_score:.1f}/100"
            )
        else:
            summary = build_candidate_summary(sc, norm)
        candidates_with_summaries.append((sc, summary))

    # Step 3 — async LLM calls (with optional cache)
    candidate_ids = [sc.candidate_id for sc in top_50]
    cache_key = _compute_cache_key(jd_text, candidate_ids)
    cached_responses: Optional[dict] = _load_llm_cache(cache_key) if use_cache else None

    if cached_responses is not None:
        print(f"  → Using cached LLM responses (key={cache_key})")
        results: list[tuple[ScoredCandidate, ReasoningCard, float, bool]] = []
        sc_map = {sc.candidate_id: sc for sc in top_50}
        for cid, resp in cached_responses.items():
            sc = sc_map.get(cid)
            if sc is None:
                continue
            card = ReasoningCard(
                strength=resp.get("strength", ""),
                gap=resp.get("gap", ""),
                recommendation=resp.get("recommendation", ""),
                llm_score=int(resp.get("llm_score", 5)),
                verified=True,
            )
            results.append((sc, card, card.llm_score * 10.0, True))
    else:
        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)
        print(f"  → Calling {_MODEL} for {len(top_50)} candidates "
              f"(semaphore={_SEMAPHORE_LIMIT})...")
        results = asyncio.run(
            rerank_all_async(candidates_with_summaries, jd_intent, semaphore, client)
        )
        if use_cache:
            responses_to_cache = {
                sc.candidate_id: {
                    "strength": card.strength,
                    "gap": card.gap,
                    "recommendation": card.recommendation,
                    "llm_score": card.llm_score,
                }
                for sc, card, _score, _ok in results
                if _ok
            }
            _save_llm_cache(cache_key, responses_to_cache)

    # Step 4 — merge scores
    for scored, reasoning, llm_score_100, api_success in results:
        merge_scores(scored, reasoning, llm_score_100, api_success=api_success)

    # Step 5 — re-sort and re-rank
    all_candidates = top_50 + scored_candidates[50:]
    all_candidates.sort(key=lambda c: c.final_score, reverse=True)
    for rank, sc in enumerate(all_candidates, start=1):
        sc.rank = rank

    # Step 6 — slice to top_k
    result = all_candidates[:top_k]

    # Step 7 — assert LLM did not influence final_score
    # Tolerance of 5.0 pts accounts for the GitHub boost (≤ ~4.9 pts at weight=0.07).
    for c in result:
        assert abs(c.final_score - c.signal_fusion_score) < 5.0, (
            f"{c.candidate_id}: final_score {c.final_score} != fusion "
            f"{c.signal_fusion_score} — LLM is still influencing ranking"
        )

    # Step 8 — log
    elapsed = time.time() - start
    log_layer_complete(layer=3, candidates_remaining=len(result), elapsed_seconds=elapsed)

    return result


# ===========================================================================
# QUICK TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer as _ST

    from pipeline.layer0_normalizer import run_layer0
    from pipeline.layer1_retrieval import run_layer1
    from pipeline.layer2_scoring import run_layer2
    from pipeline.layer2_github import run_github_enrichment

    DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

    # --- Check API key ---
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "\n  ⚠  OPENAI_API_KEY not set in .env\n"
            "     LLM calls will fail; Layer 3 will fall back to L2 scores.\n"
            "     Set OPENAI_API_KEY in .env for full LLM re-ranking.\n"
        )

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
    all_candidates_raw: list[dict] = json.loads(
        (DATA_DIR / "candidates.json").read_text(encoding="utf-8")
    )
    type_by_id: dict[str, str] = {
        c["candidate_id"]: c.get("type", "") for c in all_candidates_raw
    }

    print("\n" + "=" * 70)
    print("  Layer 3 Quick Test — jd_senior_ml.txt + 30 candidates")
    print("=" * 70 + "\n")

    # --- Layer 0 ---
    jd_intent, normalized = run_layer0(jd_text, "JD_SENIOR_ML", all_candidates_raw)
    print(f"\n  Layer 0: {len(normalized)} candidates normalised.\n")

    # --- Layer 1 ---
    retrieved, model = run_layer1(jd_intent, normalized, top_k=50)
    print(f"\n  Layer 1: {len(retrieved)} candidates retrieved.\n")

    # --- Layer 2 ---
    if model is None:
        print("  Layer 1 returned no model — loading separately...")
        model = _ST("all-MiniLM-L6-v2")
    shortlist, _hidden_gems, _low_match = run_layer2(
        jd_intent, retrieved, model, top_k=50
    )
    print(f"\n  Layer 2: {len(shortlist)} candidates scored.\n")

    # --- GitHub Enrichment ---
    candidates_map: dict[str, CandidateNormalized] = {
        c.candidate_id: c for c in normalized
    }
    print("  Running GitHub enrichment...")
    enriched = run_github_enrichment(shortlist, candidates_map, jd_intent, max_candidates=50)
    print(f"\n  GitHub: {len(enriched)} candidates enriched.\n")

    # --- Layer 3 ---
    print("  Running Layer 3 LLM re-ranking...\n")
    final_shortlist = run_layer3(
        jd_intent, enriched, candidates_map, top_k=20
    )

    # --- Results table ---
    _BORDER = "─" * 72
    print(f"\n  Final Ranking (top {len(final_shortlist)} of {len(enriched)}):")
    print(f"  {_BORDER}")
    hdr = (
        f"  {'Rank':>4} | {'ID':<5} | {'Type':<22} | "
        f"{'Fusion':>6} | {'LLM':>5} | {'Final':>6} | Conflict"
    )
    print(hdr)
    print(f"  {_BORDER}")

    for sc in final_shortlist:
        ctype = type_by_id.get(sc.candidate_id, "")
        llm_disp = f"{sc.llm_rerank_score:.0f}" if sc.llm_rerank_score is not None else "—"
        conflict_disp = "Yes" if sc.flags.score_conflict else "No"
        print(
            f"  {sc.rank:>4} | {sc.candidate_id:<5} | {ctype:<22} | "
            f"{sc.signal_fusion_score:>6.1f} | {llm_disp:>5} | "
            f"{sc.final_score:>6.1f} | {conflict_disp}"
        )

    # --- Top 3 reasoning cards ---
    _THICK = "═" * 50
    _THIN  = "─" * 50
    print(f"\n  Reasoning Cards — Top 3:")
    for sc in final_shortlist[:3]:
        ctype = type_by_id.get(sc.candidate_id, "")
        card = sc.reasoning
        print(f"\n  {_THICK}")
        print(f"  Rank {sc.rank} — {sc.candidate_id} ({ctype})")
        print(f"  Final score: {sc.final_score:.1f}/100")
        print(f"  {_THIN}")
        if card:
            print(f"  STRENGTH:       {card.strength}")
            print(f"  GAP:            {card.gap}")
            print(f"  RECOMMENDATION: {card.recommendation}")
            print(f"  LLM Score:      {card.llm_score}/10")
            print(f"  Verified:       {card.verified}")
            print(f"  Conflict:       {'Yes' if sc.flags.score_conflict else 'No'}")
        else:
            print("  (no reasoning card)")
        print(f"  {_THICK}")

    # --- Summary stats ---
    conflicts = sum(1 for sc in final_shortlist if sc.flags.score_conflict)
    verified_count = sum(
        1 for sc in final_shortlist
        if sc.flags.reasoning_verified and sc.reasoning is not None
    )
    llm_reranked = sum(1 for sc in final_shortlist if sc.flags.llm_rerank)
    elapsed_total = time.time()

    print(f"\n  Summary:")
    print(f"    Candidates re-ranked:  {len(final_shortlist)}")
    print(f"    LLM re-ranked (True):  {llm_reranked}")
    print(f"    Conflicts detected:    {conflicts}")
    print(f"    Reasoning verified:    {verified_count}/{len(final_shortlist)}")

    # --- Assertions ---
    print("\n  Assertions:")

    top1 = final_shortlist[0]
    assert top1.tier in ("strong_match", "good_match", "hidden_gem"), (
        f"Expected top candidate tier in (strong_match, good_match, hidden_gem), "
        f"got {top1.tier!r} (score={top1.final_score:.1f})"
    )
    print(f"  ✓ Top candidate is '{top1.tier}' (score={top1.final_score:.1f}).")

    top5_types = [type_by_id.get(sc.candidate_id, "") for sc in final_shortlist[:5]]
    assert "poor_match" not in top5_types, (
        f"Found poor_match in top 5: {top5_types}"
    )
    print("  ✓ No poor_match in top 5.")

    for sc in final_shortlist:
        assert sc.reasoning is not None, (
            f"{sc.candidate_id}: missing reasoning card in top_k shortlist."
        )
    print(f"  ✓ All {len(final_shortlist)} shortlisted candidates have reasoning cards.")

    # Re-sort invariant
    for i in range(len(final_shortlist) - 1):
        assert final_shortlist[i].final_score >= final_shortlist[i + 1].final_score, (
            f"Rank order violated: rank {i+1} ({final_shortlist[i].final_score:.2f}) "
            f"< rank {i+2} ({final_shortlist[i+1].final_score:.2f})"
        )
    print("  ✓ Final shortlist sorted by final_score descending.")

    print("\n  ✓ All Layer 3 assertions passed.")
