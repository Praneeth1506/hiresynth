"""
Input validation and sanitization utilities for the HireSynth pipeline.

All validation runs before any data enters Layer 0. Functions in this module
never call external APIs or import third-party libraries — only the standard
library and utils.schema are used.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import List, Tuple

from utils.schema import CandidateRaw, JDIntent


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict = {
    "skills": 0.40,
    "trajectory": 0.20,
    "recency": 0.15,
    "behavioral": 0.15,
    "seniority": 0.10,
}

# Month abbreviations and full names → zero-padded month number
_MONTH_MAP: dict = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}

_PRESENT_TOKENS = {"present", "current", "now", "ongoing", "till date", "till now"}


# ---------------------------------------------------------------------------
# 1. JD text validation
# ---------------------------------------------------------------------------

def _extract_json_strings(obj: object) -> list[str]:
    """Recursively collect all string leaf values from a JSON-decoded object."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        result: list[str] = []
        for v in obj.values():
            result.extend(_extract_json_strings(v))
        return result
    if isinstance(obj, list):
        result = []
        for item in obj:
            result.extend(_extract_json_strings(item))
        return result
    return []


def validate_jd_text(text: str) -> Tuple[bool, str]:
    """
    Validate raw job-description text before it enters the JD parser.

    If the text starts with '{' or '[', it is treated as a JSON-formatted JD
    (e.g. an ATS export or API response).  All string values are extracted
    recursively and joined so the length and word-count checks apply to the
    meaningful content rather than the JSON envelope.

    Checks:
    - Not empty or whitespace-only
    - At least 50 characters (of meaningful text)
    - Contains more than 10 words (not just a bare job title)

    Returns:
        (True, "") on success; (False, reason_string) on failure.
    """
    if not text or not text.strip():
        return False, "JD text is empty or contains only whitespace."

    stripped = text.strip()

    # JSON-formatted JD: extract string values and validate the joined content.
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
            parts = _extract_json_strings(parsed)
            stripped = " ".join(p.strip() for p in parts if p.strip())
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through and validate the raw text as-is

    if len(stripped) < 50:
        return False, (
            f"JD text is too short ({len(stripped)} chars). "
            "Minimum required: 50 characters."
        )

    word_count = len(stripped.split())
    if word_count <= 10:
        return False, (
            f"JD text has only {word_count} words — looks like a bare job title. "
            "Minimum required: more than 10 words."
        )

    return True, ""


# ---------------------------------------------------------------------------
# 2. Resume text validation
# ---------------------------------------------------------------------------

def validate_resume_text(text: str) -> Tuple[bool, str]:
    """
    Validate raw resume text before it enters the normalizer.

    Checks:
    - Not empty or whitespace-only
    - At least 30 characters
    - Not a scan-only PDF artifact (delegated to detect_scan_pdf)

    Returns:
        (True, "") on success; (False, reason_string) on failure.
    """
    if not text or not text.strip():
        return False, "Resume text is empty or contains only whitespace."

    stripped = text.strip()

    if len(stripped) < 30:
        return False, (
            f"Resume text is too short ({len(stripped)} chars). "
            "Minimum required: 30 characters."
        )

    if detect_scan_pdf(stripped):
        return False, (
            "Resume appears to be a scan-only PDF with no extractable text. "
            "Manual OCR or re-upload required."
        )

    return True, ""


# ---------------------------------------------------------------------------
# 3. Date string normalization
# ---------------------------------------------------------------------------

def normalize_date_string(date_str: str) -> str:
    """
    Normalize a free-form date string to 'YYYY-MM', 'present', or 'unknown'.

    Handled input formats:
        'Jan 2019', 'January 2019'  →  '2019-01'
        '2019-01'                   →  '2019-01'
        '01/2019'                   →  '2019-01'
        '2019'                      →  '2019-01'  (month defaults to 01)
        'Present', 'current', 'now' →  'present'

    Returns:
        Normalized date string, or 'unknown' if the format is unrecognised.
    """
    if not date_str:
        return "unknown"

    cleaned = date_str.strip().lower()

    # Present / current variants
    if cleaned in _PRESENT_TOKENS:
        return "present"

    # "Jan 2019" or "January 2019"
    match = re.fullmatch(r"([a-z]+)\s+(\d{4})", cleaned)
    if match:
        month_word, year = match.groups()
        month_num = _MONTH_MAP.get(month_word)
        if month_num:
            return f"{year}-{month_num}"

    # "2019-01" (already ISO-like) or "2019-1"
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", cleaned)
    if match:
        year, month = match.groups()
        return f"{year}-{month.zfill(2)}"

    # "01/2019" or "1/2019"
    match = re.fullmatch(r"(\d{1,2})/(\d{4})", cleaned)
    if match:
        month, year = match.groups()
        return f"{year}-{month.zfill(2)}"

    # Bare year "2019"
    match = re.fullmatch(r"(\d{4})", cleaned)
    if match:
        return f"{match.group(1)}-01"

    return "unknown"


# ---------------------------------------------------------------------------
# 4. JD parse output validation
# ---------------------------------------------------------------------------

def validate_jd_parse_output(raw: dict) -> JDIntent:
    """
    Convert a raw LLM response dict into a validated JDIntent, filling gaps
    with safe defaults.

    Handles:
    - Markdown code fences (```json ... ```) wrapping the JSON string
    - Missing or wrongly-typed fields — replaced with defaults
    - Any JSON parse error — falls back entirely to defaults

    Never raises; always returns a valid JDIntent.
    """
    # If the LLM returned a string instead of a pre-parsed dict, unwrap it.
    data: dict = {}
    if isinstance(raw, str):
        text = raw.strip()
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            data = {}
    elif isinstance(raw, dict):
        data = raw

    def _list(key: str, default: list | None = None) -> list:
        val = data.get(key, default or [])
        return val if isinstance(val, list) else []

    def _str(key: str, default: str = "") -> str:
        val = data.get(key, default)
        return str(val) if val is not None else default

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def _bool(key: str, default: bool = False) -> bool:
        val = data.get(key, default)
        return bool(val)

    weights = data.get("weights")
    if not isinstance(weights, dict):
        weights = DEFAULT_WEIGHTS.copy()

    return JDIntent(
        job_id=_str("job_id", "JD_UNKNOWN"),
        job_title=_str("job_title", "Unknown Role"),
        must_have=_list("must_have"),
        nice_have=_list("nice_have"),
        seniority=_str("seniority", "unknown"),
        domain=_str("domain", "general"),
        culture_signals=_list("culture_signals"),
        implicit_needs=_list("implicit_needs"),
        retrieval_mode=_str("retrieval_mode", "semantic"),
        weights=weights,
        quality_score=_float("quality_score", 50.0),
        quality_level=_str("quality_level", "MEDIUM"),
        is_multi_role=_bool("is_multi_role", False),
        primary_track=data.get("primary_track") if isinstance(data.get("primary_track"), dict) else None,
        secondary_track=data.get("secondary_track") if isinstance(data.get("secondary_track"), dict) else None,
    )


# ---------------------------------------------------------------------------
# 5. Unique candidate ID enforcement
# ---------------------------------------------------------------------------

def ensure_unique_candidate_ids(
    candidates: List[CandidateRaw],
) -> List[CandidateRaw]:
    """
    Guarantee that every CandidateRaw in the list has a unique, non-empty ID.

    Strategy:
    - Collect all existing IDs; track duplicates.
    - For any candidate whose ID is empty, None, or already seen:
      generate a deterministic replacement via
      md5(raw_text_prefix)[:8]  (raw text is always present).
    - If even the generated ID collides, append a counter suffix.

    Returns the same list with IDs mutated in-place (a copy of each record
    is made so callers' original objects are not modified).
    """
    seen: set = set()
    result: List[CandidateRaw] = []

    for cand in candidates:
        cid = (cand.candidate_id or "").strip()

        if not cid or cid in seen:
            # Build deterministic ID from text content
            seed = (cand.raw_text or "")[:256].encode("utf-8", errors="replace")
            cid = hashlib.md5(seed).hexdigest()[:8]

            # Handle hash collision (extremely rare, but guarded)
            if cid in seen:
                counter = 1
                while f"{cid}_{counter}" in seen:
                    counter += 1
                cid = f"{cid}_{counter}"

        seen.add(cid)
        # Return a new CandidateRaw so we don't mutate the caller's object
        result.append(CandidateRaw(
            candidate_id=cid,
            raw_text=cand.raw_text,
            source_file=cand.source_file,
        ))

    return result


# ---------------------------------------------------------------------------
# 6. Scan-PDF detection
# ---------------------------------------------------------------------------

def detect_scan_pdf(text: str) -> bool:
    """
    Heuristically determine whether text was extracted from a scan-only PDF.

    Returns True when any of the following are true:
    - Fewer than 20 real words (tokens with at least one letter or digit)
    - More than 30% of characters are non-ASCII
    - The most common non-space character accounts for > 40% of all chars
      (indicates repeated OCR artefacts / symbols)
    """
    if not text:
        return True

    # Real-word count: tokens with at least one alphanumeric character
    words = [t for t in text.split() if re.search(r"[A-Za-z0-9]", t)]
    if len(words) < 20:
        return True

    # Non-ASCII ratio
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii / len(text) > 0.30:
        return True

    # Repeated-symbol ratio: any single non-space char dominates the text
    non_space = [ch for ch in text if ch != " "]
    if non_space:
        from collections import Counter
        most_common_count = Counter(non_space).most_common(1)[0][1]
        if most_common_count / len(non_space) > 0.40:
            return True

    return False


# ---------------------------------------------------------------------------
# 7. Full pipeline input validation
# ---------------------------------------------------------------------------

def validate_pipeline_inputs(
    jd_text: str,
    candidates: List[CandidateRaw],
) -> Tuple[bool, List[str]]:
    """
    Run all pre-pipeline validations in sequence and aggregate errors.

    Checks performed (in order):
    1. JD text is valid (non-empty, long enough, not just a title)
    2. At least one candidate is provided
    3. No completely empty resume texts
    4. All candidate IDs are unique (reports duplicates, does not auto-fix)

    Returns:
        (True, [])                    — all checks passed
        (False, [error_message, …])   — one or more checks failed
    """
    errors: List[str] = []

    # 1. JD validation
    jd_ok, jd_reason = validate_jd_text(jd_text)
    if not jd_ok:
        errors.append(f"JD validation failed: {jd_reason}")

    # 2. At least one candidate
    if not candidates:
        errors.append("No candidates provided — pipeline cannot run.")
        return False, errors  # No point checking further

    # 3. Empty resume texts
    empty_resumes = [
        c.candidate_id for c in candidates
        if not (c.raw_text or "").strip()
    ]
    if empty_resumes:
        errors.append(
            f"Candidates with completely empty resume text: {empty_resumes}"
        )

    # 4. Duplicate IDs
    seen: set = set()
    duplicates: set = set()
    for c in candidates:
        cid = (c.candidate_id or "").strip()
        if cid in seen:
            duplicates.add(cid)
        seen.add(cid)
    if duplicates:
        errors.append(
            f"Duplicate candidate IDs detected (will be auto-fixed by "
            f"ensure_unique_candidate_ids): {sorted(duplicates)}"
        )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PASS = "\033[32mPASS\033[0m"
    _FAIL = "\033[31mFAIL\033[0m"

    def _check(label: str, condition: bool) -> None:
        print(f"  [{_PASS if condition else _FAIL}] {label}")

    print("\n=== validate_jd_text ===")
    ok, _ = validate_jd_text("")
    _check("empty string → False", not ok)
    ok, _ = validate_jd_text("Software Engineer")
    _check("bare title (<= 10 words, < 50 chars) → False", not ok)
    ok, _ = validate_jd_text(
        "We are looking for a Senior ML Engineer with 5+ years of experience "
        "in Python, PyTorch, and distributed training systems."
    )
    _check("valid JD text → True", ok)
    # JSON-formatted JD (ATS export)
    import json as _json
    _jd_json = _json.dumps({
        "title": "Senior ML Engineer",
        "description": "We are looking for an experienced engineer with Python and PyTorch.",
        "requirements": ["5+ years ML experience", "Strong knowledge of deep learning frameworks"],
    })
    ok, _ = validate_jd_text(_jd_json)
    _check("JSON-formatted JD with enough content → True", ok)
    _jd_json_short = _json.dumps({"title": "Engineer"})
    ok, _ = validate_jd_text(_jd_json_short)
    _check("JSON-formatted JD with too-short content → False", not ok)

    print("\n=== validate_resume_text ===")
    ok, _ = validate_resume_text("   ")
    _check("whitespace only → False", not ok)
    ok, _ = validate_resume_text("John Doe")
    _check("too short → False", not ok)
    ok, _ = validate_resume_text(
        "John Doe is a Senior ML Engineer with five years of hands-on experience "
        "building and deploying large-scale deep learning systems at Google and Meta. "
        "Proficient in Python, PyTorch, and distributed training on TPU clusters."
    )
    _check("valid resume text → True", ok)
    scan_garbage = "\x00\xff\xfe" * 50 + "garbage"
    ok, _ = validate_resume_text(scan_garbage)
    _check("scan-PDF garbage → False", not ok)

    print("\n=== normalize_date_string ===")
    _check('"Jan 2019" → "2019-01"', normalize_date_string("Jan 2019") == "2019-01")
    _check('"January 2019" → "2019-01"', normalize_date_string("January 2019") == "2019-01")
    _check('"2019-01" → "2019-01"', normalize_date_string("2019-01") == "2019-01")
    _check('"01/2019" → "2019-01"', normalize_date_string("01/2019") == "2019-01")
    _check('"2019" → "2019-01"', normalize_date_string("2019") == "2019-01")
    _check('"Present" → "present"', normalize_date_string("Present") == "present")
    _check('"current" → "present"', normalize_date_string("current") == "present")
    _check('"now" → "present"', normalize_date_string("now") == "present")
    _check('"gibberish" → "unknown"', normalize_date_string("gibberish") == "unknown")

    print("\n=== validate_jd_parse_output ===")
    # Full valid dict
    valid_raw = {
        "job_id": "JD_001", "job_title": "ML Engineer",
        "must_have": ["Python"], "nice_have": ["Kubernetes"],
        "seniority": "senior", "domain": "ml",
        "culture_signals": ["fast-paced"], "implicit_needs": ["ownership"],
        "retrieval_mode": "technical", "weights": {"skills": 0.4},
        "quality_score": 85, "quality_level": "HIGH",
        "is_multi_role": False,
    }
    jd = validate_jd_parse_output(valid_raw)
    _check("valid dict → JDIntent with correct fields",
           jd.job_title == "ML Engineer" and jd.quality_score == 85.0)

    # Empty dict → all defaults
    jd2 = validate_jd_parse_output({})
    _check("empty dict → safe defaults",
           jd2.seniority == "unknown" and jd2.domain == "general"
           and jd2.weights == DEFAULT_WEIGHTS)

    # Markdown-fenced JSON string
    fenced = '```json\n{"job_id": "JD_X", "job_title": "Data Scientist"}\n```'
    jd3 = validate_jd_parse_output(fenced)  # type: ignore[arg-type]
    _check("markdown-fenced JSON string → parsed correctly",
           jd3.job_title == "Data Scientist")

    print("\n=== ensure_unique_candidate_ids ===")
    dupes = [
        CandidateRaw("C001", "Alice resume text about ML engineering experience"),
        CandidateRaw("C001", "Bob resume text about data science and Python work"),
        CandidateRaw("", "Carol resume text about devops kubernetes cloud"),
    ]
    fixed = ensure_unique_candidate_ids(dupes)
    ids = [c.candidate_id for c in fixed]
    _check("all IDs unique after fix", len(ids) == len(set(ids)))
    _check("first valid ID preserved", ids[0] == "C001")
    _check("no empty IDs remain", all(cid for cid in ids))

    print("\n=== detect_scan_pdf ===")
    _check("normal text → False",
           not detect_scan_pdf(
               "John Doe Senior ML Engineer Python PyTorch TensorFlow "
               "Google Brain 5 years experience deep learning NLP computer vision "
               "distributed systems Kubernetes Docker cloud AWS GCP Azure."
           ))
    _check("too few words → True",
           detect_scan_pdf("abc def"))
    _check("mostly non-ASCII → True",
           detect_scan_pdf("\xff\xfe\xfd" * 40 + " word"))

    print("\n=== validate_pipeline_inputs ===")
    good_jd = (
        "We are hiring a Senior ML Engineer with expertise in Python, PyTorch, "
        "and large-scale distributed training. The role involves designing and "
        "deploying production ML systems."
    )
    good_candidates = [
        CandidateRaw("C001", "Alice has 6 years of ML experience building recommendation "
                              "systems at scale using Python and PyTorch at top tech firms."),
        CandidateRaw("C002", "Bob specializes in NLP and has shipped BERT-based pipelines "
                              "to production serving millions of daily active users."),
    ]
    ok, errs = validate_pipeline_inputs(good_jd, good_candidates)
    _check("valid inputs → (True, [])", ok and errs == [])

    ok, errs = validate_pipeline_inputs("", good_candidates)
    _check("empty JD → (False, [error])", not ok and len(errs) == 1)

    ok, errs = validate_pipeline_inputs(good_jd, [])
    _check("no candidates → (False, [error])", not ok)

    dup_candidates = [
        CandidateRaw("C001", "Alice resume text about machine learning engineering work"),
        CandidateRaw("C001", "Bob resume text about data science Python TensorFlow work"),
    ]
    ok, errs = validate_pipeline_inputs(good_jd, dup_candidates)
    _check("duplicate IDs → (False, [error])", not ok and any("Duplicate" in e for e in errs))

    print()
