"""
pipeline/layer2_github.py — GitHub Enrichment (between Layer 2 and Layer 3).

Enriches the top 50 scored candidates with signals derived from the GitHub
API (profile stats, repository metadata, push recency).  This is a
*boost-only* signal: candidates without a GitHub profile or with a 404 /
timeout response are never penalised — their final_score is unchanged.

The enrichment step:
  1. Extracts the GitHub username from the candidate's github_url field.
  2. Calls three GitHub REST API endpoints with a hard 2-second timeout each.
  3. Computes a composite github_score (0.0–1.0) from five sub-signals.
  4. Adds a weighted boost (≈ 7%) to final_score for verified profiles.
  5. Re-sorts and re-ranks all returned candidates by updated final_score.

Sub-signals and weights:
  0.30 — Recency       (days since last PushEvent)
  0.20 — Volume        (public_repos count)
  0.20 — Credibility   (total stars across all repos)
  0.20 — Language match(top-3 normalised languages vs JD must-have)
  0.10 — Account age   (years since account creation)

API rate limits:
  — Unauthenticated:  60 requests / hour
  — Authenticated:  5000 requests / hour  (set GITHUB_TOKEN in .env)

Usage:
    python -m pipeline.layer2_github
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm

from utils.logger import log_layer_complete, log_warning
from utils.schema import (
    CandidateNormalized,
    GitHubProfile,
    JDIntent,
    ScoredCandidate,
    SignalBreakdown,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GITHUB_API_BASE = "https://api.github.com"
_TIMEOUT = 2  # hard timeout per API call (seconds)

# Language normalisation: map repo language → canonical skill name used by JDs.
LANG_MAP: dict[str, str] = {
    "Jupyter Notebook": "Python",
    "TypeScript":       "JavaScript",
    "C++":              "C",
    "Shell":            "Bash",
    "Dockerfile":       "Docker",
}


# ===========================================================================
# PART 1 — GITHUB API CLIENT
# ===========================================================================

def get_github_headers() -> dict:
    """
    Build request headers for the GitHub REST API v3.

    Uses Bearer-token authentication when GITHUB_TOKEN is present in the
    environment (5,000 req/hr), falling back to unauthenticated headers
    (60 req/hr) when the variable is absent or empty.
    """
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
    return {"Accept": "application/vnd.github.v3+json"}


def fetch_github_user(username: str) -> Optional[dict]:
    """
    Fetch the GitHub user-profile object for a given username.

    Endpoint: GET /users/{username}

    Returns:
        Parsed JSON dict on HTTP 200; None on 404 (profile not found),
        403 (rate limit exceeded), request timeout, or any other error.
    """
    url = f"{_GITHUB_API_BASE}/users/{username}"
    try:
        resp = requests.get(url, headers=get_github_headers(), timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if resp.status_code == 403:
            log_warning(f"GitHub rate limit hit fetching user '{username}' (403).")
            return None
        log_warning(
            f"GitHub /users/{username} returned HTTP {resp.status_code} — skipping."
        )
        return None
    except requests.Timeout:
        log_warning(
            f"GitHub /users/{username} timed out after {_TIMEOUT}s — skipping."
        )
        return None
    except requests.RequestException as exc:
        log_warning(f"GitHub /users/{username} request error: {exc}")
        return None


def fetch_github_events(username: str) -> list[dict]:
    """
    Fetch the public event stream for a given username.

    Endpoint: GET /users/{username}/events

    Events are ordered newest-first by the GitHub API; the first PushEvent
    in the list is therefore the most recent push.

    Returns:
        List of event dicts on HTTP 200; empty list on any failure.
    """
    url = f"{_GITHUB_API_BASE}/users/{username}/events"
    try:
        resp = requests.get(url, headers=get_github_headers(), timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        return []
    except (requests.Timeout, requests.RequestException):
        return []


def fetch_github_repos(username: str, max_repos: int = 30) -> list[dict]:
    """
    Fetch the most recently updated public repositories for a given username.

    Endpoint: GET /users/{username}/repos
    Query params: sort=updated, per_page=max_repos

    Returns:
        List of repository dicts on HTTP 200; empty list on any failure.
    """
    url = f"{_GITHUB_API_BASE}/users/{username}/repos"
    params: dict = {"sort": "updated", "per_page": max_repos}
    try:
        resp = requests.get(
            url, headers=get_github_headers(), params=params, timeout=_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json()
        return []
    except (requests.Timeout, requests.RequestException):
        return []


# ===========================================================================
# PART 2 — USERNAME EXTRACTION
# ===========================================================================

def extract_github_username(github_url: Optional[str]) -> Optional[str]:
    """
    Parse a GitHub profile URL or handle and return the bare username string.

    Handles the following input formats:
      — "https://github.com/username"
      — "github.com/username"
      — "github.com/username/"    (trailing slash stripped)
      — "@username"               (leading @ treated as GitHub handle)

    Returns:
        Username string on success (no scheme, no slash, no query params);
        None if the input is None, empty, or cannot be parsed.
    """
    if not github_url:
        return None

    url = github_url.strip()

    # Handle @username shorthand (assume GitHub)
    if url.startswith("@"):
        candidate = url[1:].split("?")[0].split("#")[0].split("/")[0].strip()
        return candidate or None

    # Strip scheme (http:// or https://)
    url = re.sub(r"^https?://", "", url)

    # Strip query string and fragment, then trailing slashes
    url = url.split("?")[0].split("#")[0].rstrip("/")

    # Expect github.com/{username}[/{anything}]
    if url.startswith("github.com/"):
        parts = url.split("/")
        # parts[0] = "github.com", parts[1] = username
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return None

    # Plain identifier with no domain — treat as username directly
    if "/" not in url and url:
        return url

    return None


# ===========================================================================
# PART 3 — SCORING
# ===========================================================================

def compute_github_score(
    user_data: dict,
    events: list[dict],
    repos: list[dict],
    jd_intent: JDIntent,
) -> tuple[float, GitHubProfile]:
    """
    Compute a composite GitHub score (0.0–1.0) from five weighted sub-signals.

    Sub-signals (weights sum to 1.0):
      0.30 — Recency      days since most recent PushEvent
      0.20 — Volume       public_repos count
      0.20 — Credibility  total stars across all repos
      0.20 — Language match top-3 normalised languages vs JD must-have skills
      0.10 — Account age  years since account creation

    Language normalisation: LANG_MAP is applied before matching so that
    e.g. "Jupyter Notebook" is treated as "Python".
    Language match uses case-insensitive substring containment against the
    joined JD must-have skill phrases.

    Returns:
        (github_score, GitHubProfile) where github_score is in [0.0, 1.0].
    """
    now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ 1. Recency
    last_push_date: Optional[str] = None
    recency_score = 0.3  # default when no push events are present

    for event in events:
        if event.get("type") == "PushEvent":
            raw_ts = event.get("created_at", "")
            if raw_ts:
                last_push_date = raw_ts[:10]  # "YYYY-MM-DD"
                break  # events are newest-first; first PushEvent wins

    if last_push_date:
        try:
            push_dt = datetime.fromisoformat(last_push_date)
            push_dt = push_dt.replace(tzinfo=timezone.utc)
            days_since = (now - push_dt).days
        except ValueError:
            days_since = 9999

        if days_since <= 30:
            recency_score = 1.0
        elif days_since <= 90:
            recency_score = 0.8
        elif days_since <= 180:
            recency_score = 0.6
        elif days_since <= 365:
            recency_score = 0.4
        else:
            recency_score = 0.2

    # ------------------------------------------------------------------ 2. Volume
    public_repos = int(user_data.get("public_repos", 0))
    if public_repos >= 50:
        volume_score = 1.0
    elif public_repos >= 20:
        volume_score = 0.8
    elif public_repos >= 10:
        volume_score = 0.6
    elif public_repos >= 5:
        volume_score = 0.4
    else:
        volume_score = 0.2

    # -------------------------------------------------------------- 3. Credibility
    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    if total_stars >= 100:
        credibility_score = 1.0
    elif total_stars >= 50:
        credibility_score = 0.8
    elif total_stars >= 20:
        credibility_score = 0.6
    elif total_stars >= 5:
        credibility_score = 0.4
    else:
        credibility_score = 0.2

    # ----------------------------------------------------------- 4. Language match
    lang_counter: Counter[str] = Counter()
    for repo in repos:
        lang = repo.get("language")
        if lang:
            normalised = LANG_MAP.get(lang, lang)
            lang_counter[normalised] += 1

    top_languages: list[str] = [lang for lang, _ in lang_counter.most_common(3)]

    if not repos or not top_languages:
        lang_match_score = 0.0
    else:
        jd_must_lower = " ".join(jd_intent.must_have).lower()
        matches = sum(
            1 for lang in top_languages
            if lang.lower() in jd_must_lower
        )
        lang_match_score = min(matches / 3.0, 1.0)

    # ------------------------------------------------------------ 5. Account age
    created_at_str: str = user_data.get("created_at", "")
    account_age_years: Optional[float] = None

    if created_at_str:
        try:
            created_dt = datetime.fromisoformat(
                created_at_str.replace("Z", "+00:00")
            )
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            account_age_years = (now - created_dt).days / 365.0
        except ValueError:
            pass

    if account_age_years is None:
        age_score = 0.2
    elif account_age_years >= 5:
        age_score = 1.0
    elif account_age_years >= 3:
        age_score = 0.8
    elif account_age_years >= 1:
        age_score = 0.5
    else:
        age_score = 0.2

    # -------------------------------------------------------------- Composite
    github_score = round(
        0.30 * recency_score
        + 0.20 * volume_score
        + 0.20 * credibility_score
        + 0.20 * lang_match_score
        + 0.10 * age_score,
        6,
    )

    profile = GitHubProfile(
        username=str(user_data.get("login", "")),
        public_repos=public_repos,
        total_stars=total_stars,
        top_languages=top_languages,
        last_push_date=last_push_date,
        account_age_years=account_age_years,
        followers=int(user_data.get("followers", 0)),
        github_score=github_score,
        verified=True,
    )

    return github_score, profile


# ===========================================================================
# PART 4 — SIGNAL INTEGRATION
# ===========================================================================

def apply_github_boost(
    scored: ScoredCandidate,
    github_score: float,
    github_profile: GitHubProfile,
    github_weight: float = 0.07,
) -> ScoredCandidate:
    """
    Add a boost-only GitHub contribution to a ScoredCandidate's final_score.

    Boost formula: boost_pts = github_score × github_weight × 100
    New score   : min(100.0, old_final_score + boost_pts)

    The operation is purely additive — a github_score of 0 contributes 0 pts.
    Absence of a GitHub profile (github_verified=False) means this function
    is never called, so unverified candidates are never touched.

    Mutates the ScoredCandidate in-place and returns it for convenience.
    """
    boost = round(github_score * github_weight * 100, 4)
    scored.final_score = min(100.0, round(scored.final_score + boost, 2))

    scored.signal_breakdown["github_boost"] = SignalBreakdown(
        raw=github_score,
        normalized=github_score,
        weighted=boost,
    )
    scored.flags.github_verified = True
    scored.github_profile = github_profile

    return scored


# ===========================================================================
# PART 5 — ORCHESTRATOR
# ===========================================================================

def run_github_enrichment(
    scored_candidates: list[ScoredCandidate],
    candidates_map: dict[str, CandidateNormalized],
    jd_intent: JDIntent,
    max_candidates: int = 50,
) -> list[ScoredCandidate]:
    """
    Run GitHub enrichment for the top max_candidates scored candidates.

    Per-candidate pipeline:
      1. Look up the CandidateNormalized record in candidates_map.
      2. Extract the GitHub username from github_url.
      3. If no username → mark github_verified=False and continue.
      4. Fetch user profile (GET /users/{username}).
      5. If None (404 / timeout / error) → mark github_verified=False and continue.
      6. Fetch events (GET /users/{username}/events).
      7. Fetch repos  (GET /users/{username}/repos).
      8. Compute composite github_score and build GitHubProfile.
      9. Apply boost via apply_github_boost() — mutates final_score.

    After all enrichments:
      — The full list (top + rest) is re-sorted by final_score descending.
      — Ranks are re-assigned starting from 1.
      — Completion is logged with log_layer_complete.

    Returns:
        Complete list of ScoredCandidate objects (all input candidates, not
        just the top max_candidates), re-sorted and re-ranked.
    """
    start = time.time()

    top_candidates = scored_candidates[:max_candidates]
    rest = scored_candidates[max_candidates:]

    verified_count = 0
    skipped_count = 0

    for sc in tqdm(top_candidates, desc="  GitHub enrichment", unit="cand"):
        norm = candidates_map.get(sc.candidate_id)
        github_url = norm.github_url if norm else None

        username = extract_github_username(github_url)
        if not username:
            sc.flags.github_verified = False
            skipped_count += 1
            continue

        user_data = fetch_github_user(username)
        if user_data is None:
            sc.flags.github_verified = False
            skipped_count += 1
            continue

        events = fetch_github_events(username)
        repos = fetch_github_repos(username)

        github_score, profile = compute_github_score(
            user_data, events, repos, jd_intent
        )
        apply_github_boost(sc, github_score, profile)
        verified_count += 1

    # Re-sort entire list by updated final_score and re-assign ranks
    all_candidates = top_candidates + rest
    all_candidates.sort(key=lambda c: c.final_score, reverse=True)
    for rank, sc in enumerate(all_candidates, start=1):
        sc.rank = rank

    elapsed = time.time() - start
    print(
        f"  GitHub enrichment: {verified_count} verified, "
        f"{skipped_count} skipped (no URL / 404 / timeout)"
    )
    log_layer_complete(
        layer=2,
        candidates_remaining=len(all_candidates),
        elapsed_seconds=elapsed,
    )

    return all_candidates


# ===========================================================================
# QUICK TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer as _ST

    from pipeline.layer0_normalizer import run_layer0
    from pipeline.layer1_retrieval import run_layer1
    from pipeline.layer2_scoring import run_layer2

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
    all_candidates_raw: list[dict] = json.loads(
        (DATA_DIR / "candidates.json").read_text(encoding="utf-8")
    )
    github_url_by_id: dict[str, Optional[str]] = {
        c["candidate_id"]: c.get("github_url") for c in all_candidates_raw
    }

    print("\n" + "=" * 70)
    print("  GitHub Enrichment Quick Test — jd_senior_ml.txt + 30 candidates")
    print("=" * 70 + "\n")
    print("  Note: synthetic GitHub URLs are fake — all API calls will 404.")
    print("  Test verifies graceful 404 handling, not enrichment scores.\n")

    # --- Layers 0, 1, 2 ---
    jd_intent, normalized = run_layer0(jd_text, "JD_SENIOR_ML", all_candidates_raw)
    print(f"\n  Layer 0: {len(normalized)} candidates normalised.\n")

    retrieved = run_layer1(jd_intent, normalized, top_k=50)
    print(f"\n  Layer 1: {len(retrieved)} candidates retrieved.\n")

    print("  Loading sentence-transformer for Layer 2...")
    model = _ST("all-MiniLM-L6-v2")

    shortlist, _hidden_gems, _low_match = run_layer2(
        jd_intent, retrieved, model, top_k=50
    )
    print(f"\n  Layer 2: {len(shortlist)} candidates scored.\n")

    # Snapshot pre-enrichment scores for comparison
    pre_scores: dict[str, float] = {sc.candidate_id: sc.final_score for sc in shortlist}

    # --- GitHub enrichment ---
    candidates_map: dict[str, CandidateNormalized] = {
        c.candidate_id: c for c in normalized
    }

    print("\n  Running GitHub enrichment...\n")
    enriched = run_github_enrichment(shortlist, candidates_map, jd_intent, max_candidates=50)

    # --- Results table ---
    print("\n  GitHub Enrichment Results:")
    hdr = (
        f"  {'ID':<6} | {'URL present':^12} | {'username':^22} | "
        f"{'verified':^8} | {'gh_score':>8} | {'boost':>9}"
    )
    print("  " + "─" * (len(hdr) - 2))
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for sc in enriched:
        raw_url = github_url_by_id.get(sc.candidate_id)
        has_url = "yes" if raw_url else "no"
        username = sc.github_profile.username if sc.github_profile else "—"
        verified = sc.flags.github_verified
        gh_score = sc.github_profile.github_score if sc.github_profile else 0.0
        boost_bd = sc.signal_breakdown.get("github_boost")
        boost_str = f"+{boost_bd.weighted:.2f}pts" if boost_bd else "skipped"
        print(
            f"  {sc.candidate_id:<6} | {has_url:^12} | {username:^22} | "
            f"{'True' if verified else 'False':^8} | {gh_score:>8.3f} | {boost_str:>9}"
        )

    # --- Assertions ---
    print("\n  Assertions:")

    verified_list = [sc for sc in enriched if sc.flags.github_verified]
    unverified_list = [sc for sc in enriched if not sc.flags.github_verified]
    print(
        f"  (Info: {len(verified_list)} verified profiles, "
        f"{len(unverified_list)} skipped/404)"
    )

    # 1. Every verified candidate has a GitHubProfile and a github_boost entry
    for sc in verified_list:
        assert sc.github_profile is not None, (
            f"{sc.candidate_id}: github_verified=True but github_profile is None."
        )
        assert "github_boost" in sc.signal_breakdown, (
            f"{sc.candidate_id}: github_verified=True but no 'github_boost' in breakdown."
        )
        assert sc.signal_breakdown["github_boost"].weighted > 0, (
            f"{sc.candidate_id}: github_boost.weighted should be > 0 for verified."
        )
    print(
        f"  ✓ All {len(verified_list)} verified candidates have GitHubProfile + boost."
    )

    # 2. Unverified candidates have no boost and unchanged final_score
    for sc in unverified_list:
        assert "github_boost" not in sc.signal_breakdown, (
            f"{sc.candidate_id}: github_verified=False but 'github_boost' found in breakdown."
        )
        pre = pre_scores.get(sc.candidate_id)
        if pre is not None:
            assert abs(sc.final_score - pre) < 0.001, (
                f"{sc.candidate_id}: score changed from {pre:.2f} → {sc.final_score:.2f} "
                f"without GitHub verification."
            )
    print(
        f"  ✓ All {len(unverified_list)} unverified candidates: no boost, scores unchanged."
    )

    # 3. 404 / timeout path: no exception was raised (reaching this line proves it)
    print("  ✓ No exception raised for any candidate (404s handled gracefully).")

    # 4. Re-sort invariant: rank n score ≥ rank n+1 score
    for i in range(len(enriched) - 1):
        assert enriched[i].final_score >= enriched[i + 1].final_score, (
            f"Rank order violated: rank {i+1} score {enriched[i].final_score:.2f} "
            f"< rank {i+2} score {enriched[i+1].final_score:.2f}"
        )
    print("  ✓ Candidates re-sorted by final_score descending after enrichment.")

    print("\n  ✓ All GitHub enrichment assertions passed.")
