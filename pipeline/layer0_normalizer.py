"""
pipeline/layer0_normalizer.py — Layer 0: JD Parsing and Resume Normalization.

Two parallel streams:

  LEFT  — JD Parser (parse_jd):
      Validates JD text, scores quality 0-100, calls GPT-4o-mini to extract
      a structured JDIntent, detects multi-role JDs, and applies skill
      fallbacks when quality is LOW.

  RIGHT — Resume Normalizer (normalize_resume):
      Pass 1: rule-based extraction (regex + optional spaCy NER).
      Pass 2: GPT-4o-mini fallback for missing/unstructured fields.
      Computes trajectory_label, total_years_exp, fresher/career-changer
      flags, and confidence level.

Plus:
  dedup_candidates — cosine-similarity + metadata dedup filter.
  run_layer0       — top-level orchestrator that sequences both streams.

Usage:
    python pipeline/layer0_normalizer.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import difflib
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logger import (
    generate_run_id,
    log_layer_complete,
    log_pipeline_start,
    log_warning,
)
from utils.schema import (
    CandidateNormalized,
    CandidateRaw,
    ExperienceEntry,
    JDIntent,
)
from utils.validators import (
    DEFAULT_WEIGHTS,
    detect_scan_pdf,
    ensure_unique_candidate_ids,
    normalize_date_string,
    validate_jd_parse_output,
    validate_jd_text,
    validate_pipeline_inputs,
    validate_resume_text,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 50+ tech terms scanned across full resume text during Pass 1 skill extraction
_TECH_KEYWORDS: frozenset[str] = frozenset({
    "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "plotly",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "docker", "kubernetes", "k8s", "airflow", "mlflow", "kubeflow", "feast",
    "aws", "gcp", "azure", "sagemaker", "s3", "ec2", "cloud",
    "spark", "hadoop", "kafka", "databricks", "dbt", "hive",
    "git", "github", "gitlab", "ci/cd", "jenkins", "terraform", "ansible",
    "bert", "gpt", "llm", "transformers", "hugging face", "langchain", "rag",
    "fastapi", "flask", "django", "rest", "graphql", "grpc",
    "java", "c++", "scala", "julia", "rust", "golang",
    "react", "javascript", "typescript", "nodejs", "node.js",
    "linux", "bash", "shell",
    "xgboost", "lightgbm", "catboost",
    "computer vision", "nlp", "natural language processing",
    "deep learning", "machine learning", "reinforcement learning",
    "neural network", "cnn", "rnn", "lstm", "transformer",
    "weights & biases", "wandb", "ray", "triton", "onnx", "jax",
    "a/b testing", "feature engineering", "model serving",
})

# Canonical display names for tech keywords (lowercase key → display value)
_KEYWORD_DISPLAY: dict[str, str] = {
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn", "numpy": "NumPy", "pandas": "Pandas", "scipy": "SciPy",
    "matplotlib": "Matplotlib", "seaborn": "Seaborn", "plotly": "Plotly",
    "postgresql": "PostgreSQL", "mongodb": "MongoDB", "redis": "Redis",
    "elasticsearch": "Elasticsearch", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "airflow": "Airflow", "mlflow": "MLflow", "kubeflow": "Kubeflow", "feast": "Feast",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure", "sagemaker": "SageMaker",
    "databricks": "Databricks", "kafka": "Kafka", "spark": "Spark",
    "terraform": "Terraform", "jenkins": "Jenkins",
    "bert": "BERT", "gpt": "GPT", "llm": "LLMs", "langchain": "LangChain",
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django",
    "javascript": "JavaScript", "typescript": "TypeScript", "nodejs": "Node.js",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "nlp": "NLP", "jax": "JAX", "triton": "Triton", "onnx": "ONNX",
    "weights & biases": "Weights & Biases", "wandb": "Weights & Biases",
}

# Job title → seniority level (1=junior … 5=executive)
_LEVEL_PATTERNS: list[tuple[int, list[str]]] = [
    (1, ["intern", "trainee", "apprentice", "student", "jr ", "junior", "entry"]),
    (5, ["director", "vp ", "vice president", "cto", "ceo", "coo", "head of", "chief"]),
    (4, ["principal", "distinguished", "fellow", "partner"]),
    (4, ["lead ", "staff ", "manager", "senior manager", "group manager"]),
    (3, ["senior", "sr ", "sr.", "specialist", "consultant", "iii", "level iii"]),
]

# Fallback skill sets for low-quality JDs, keyed by domain keyword
FALLBACK_SKILLS: dict[str, list[str]] = {
    "machine_learning": ["Python", "ML", "PyTorch", "TensorFlow", "Scikit-learn"],
    "data_science":     ["Python", "SQL", "Statistics", "Pandas", "Data Analysis"],
    "mlops":            ["Docker", "Kubernetes", "CI/CD", "MLflow", "Python"],
    "general":          ["Communication", "Problem Solving", "Teamwork"],
}

# Known Indian + major global cities for location extraction
_CITIES: frozenset[str] = frozenset({
    "bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai",
    "pune", "kolkata", "ahmedabad", "noida", "gurgaon", "gurugram",
    "jaipur", "surat", "kochi", "bhubaneswar", "indore", "chandigarh",
    "san francisco", "new york", "seattle", "austin", "boston", "chicago",
    "los angeles", "london", "singapore", "toronto", "berlin", "amsterdam",
    "remote", "work from home", "wfh",
})

# Regex patterns
_GITHUB_RE = re.compile(r'https?://github\.com/([\w\-]+)', re.IGNORECASE)
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.IGNORECASE)
_DATE_RANGE_RE = re.compile(
    r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}'
    r'|\b\d{4}[-/]\d{1,2}'
    r'|\b\d{4})'
    r'\s*[-–—to]+\s*'
    r'(present|current|now|ongoing|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}'
    r'|\b\d{4}[-/]\d{1,2}'
    r'|\b\d{4})',
    re.IGNORECASE,
)
_SECTION_BOUNDARY_RE = re.compile(
    r'^(skills?|technical skills?|technologies|stack|tools?|experience|'
    r'work\s+(?:experience|history)|employment|education|academic|'
    r'projects?|certifications?|summary|objective|publications?|awards?|'
    r'contact|references?|languages?|interests?)\s*:?\s*$',
    re.IGNORECASE | re.MULTILINE,
)
_EXP_PIPE_RE = re.compile(
    r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*[-–—]\s*(.+?)\s*$',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Lazy OpenAI client
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazily create and cache the OpenAI client using the env API key."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or len(api_key) < 20:
            raise ValueError(
                "Valid OPENAI_API_KEY not found. "
                "Add it to .env: OPENAI_API_KEY=sk-..."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# LLM helpers (temperature=0 throughout; retry on transient errors)
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Call GPT-4o-mini at temperature=0 and return the raw text content.

    Retries up to 3 times with exponential back-off on transient failures.
    Raises on the final failure.
    """
    for attempt in range(3):
        try:
            resp = _get_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"    ⚠ LLM error (attempt {attempt + 1}/3): {exc} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise
    return ""


def _parse_llm_json(text: str) -> dict | list:
    """
    Strip markdown fences if present and JSON-parse LLM output.

    Returns an empty dict on any parse failure so callers never crash.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}


# ===========================================================================
# STREAM 1 — JD PARSER
# ===========================================================================

def _score_jd_quality(jd_text: str) -> tuple[float, str]:
    """
    Score JD completeness 0-100 by counting six present signal types.

    Signal weights:
      has_required_skills  +20
      has_experience_years +20
      has_seniority_level  +15
      has_domain           +15
      has_nice_to_haves    +15
      has_culture_signals  +15

    Quality levels: HIGH (>=70), MEDIUM (40-69), LOW (<40).
    """
    text = jd_text.lower()
    score = 0.0

    # has_required_skills
    skill_kws = {"python", "pytorch", "tensorflow", "docker", "kubernetes",
                 "sql", "machine learning", "deep learning", "data"}
    if any(kw in text for kw in skill_kws) or re.search(
        r"(required?|must.have|skills?\s*:|technologies?)", text
    ):
        score += 20

    # has_experience_years
    if re.search(r"\d+\+?\s*years?", text):
        score += 20

    # has_seniority_level
    if re.search(
        r"\b(senior|junior|lead|mid.?level|principal|staff|entry.?level|ic\d)\b", text
    ):
        score += 15

    # has_domain
    if re.search(
        r"\b(machine.learning|data.science|mlops|nlp|computer.vision|"
        r"deep.learning|artificial.intelligence|analytics)\b",
        text,
    ):
        score += 15

    # has_nice_to_haves
    if re.search(r"(nice.to.have|preferred|bonus|plus|advantageous|good.to.have)", text):
        score += 15

    # has_culture_signals
    if re.search(r"(fast.?paced|startup|collaborative|ownership|impact|mission|culture)", text):
        score += 15

    score = min(100.0, score)
    level = "HIGH" if score >= 70 else ("MEDIUM" if score >= 40 else "LOW")
    return score, level


def _detect_multi_role(must_have: list[str]) -> tuple[bool, dict | None, dict | None]:
    """
    Return (is_multi_role, primary_track, secondary_track).

    A JD is multi-role when must_have contains both hands-on technical
    and people-management requirements.  Primary track = technical skills;
    secondary track = managerial skills.
    """
    technical_kws = {
        "python", "pytorch", "tensorflow", "docker", "kubernetes", "sql",
        "ml", "ai", "code", "coding", "programming", "model", "algorithm",
        "data", "cloud", "aws", "gcp", "azure",
    }
    managerial_kws = {
        "manage", "lead", "mentor", "team", "people", "hire", "hiring",
        "strategy", "budget", "stakeholder", "roadmap", "cross-functional",
    }

    tech_items, mgmt_items = [], []
    for skill in must_have:
        lower = skill.lower()
        if any(kw in lower for kw in technical_kws):
            tech_items.append(skill)
        if any(kw in lower for kw in managerial_kws):
            mgmt_items.append(skill)

    if tech_items and mgmt_items:
        return True, {"skills": tech_items}, {"skills": mgmt_items}
    return False, None, None


def parse_jd(jd_text: str, job_id: str) -> JDIntent:
    """
    Parse a raw job-description string into a validated JDIntent.

    Steps:
      1. Validate JD text length — return safe default on failure.
      2. Score quality (0-100) using signal heuristics.
      3. Call GPT-4o-mini to extract structured intent JSON.
      4. Validate and fill defaults via validate_jd_parse_output.
      5. Detect multi-role JDs.
      6. If quality LOW: merge FALLBACK_SKILLS for detected domain.
    """
    # Step 1 — validate
    ok, reason = validate_jd_text(jd_text)
    if not ok:
        log_warning(f"JD validation failed ({reason}) — returning default JDIntent.")
        return JDIntent(
            job_id=job_id, job_title="Unknown Role",
            must_have=[], nice_have=[], seniority="unknown",
            domain="general", culture_signals=[], implicit_needs=[],
            retrieval_mode="semantic", weights=DEFAULT_WEIGHTS.copy(),
            quality_score=0.0, quality_level="LOW",
        )

    # Step 2 — quality score
    quality_score, quality_level = _score_jd_quality(jd_text)

    # Step 3 — LLM extraction
    system_prompt = (
        "You are a JD parser. Extract structured info from job descriptions. "
        "Return valid JSON only. No markdown fences. No explanation."
    )
    user_prompt = (
        "Parse this job description and return JSON:\n"
        "{\n"
        '  "job_title": "extracted job title",\n'
        '  "must_have": ["list of required skills/qualifications"],\n'
        '  "nice_have": ["list of preferred skills"],\n'
        '  "seniority": "junior|mid|senior|lead|any",\n'
        '  "domain": "primary domain (e.g. machine_learning, data_science, mlops)",\n'
        '  "culture_signals": ["culture/values keywords"],\n'
        '  "implicit_needs": ["inferred needs not stated explicitly"],\n'
        '  "retrieval_mode": "technical|semantic",\n'
        '  "weights": {\n'
        '    "skills": float,\n'
        '    "trajectory": float,\n'
        '    "recency": float,\n'
        '    "behavioral": float,\n'
        '    "seniority": float\n'
        "  }\n"
        "}\n\n"
        "For weights: all five must sum exactly to 1.0.\n"
        "Technical/IC roles → increase skills weight.\n"
        "Leadership roles → increase trajectory weight.\n\n"
        f"JD TEXT:\n{jd_text[:3000]}"
    )

    raw_response = _call_llm(system_prompt, user_prompt, max_tokens=1024)

    # Step 4 — validate and fill defaults
    parsed = _parse_llm_json(raw_response)
    intent = validate_jd_parse_output(parsed)

    # Overwrite computed fields
    intent.job_id = job_id
    intent.quality_score = quality_score
    intent.quality_level = quality_level

    # Step 5 — multi-role detection
    is_multi, primary, secondary = _detect_multi_role(intent.must_have)
    intent.is_multi_role = is_multi
    intent.primary_track = primary
    intent.secondary_track = secondary

    # Normalize weights to sum = 1.0
    w = intent.weights
    total = sum(w.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        intent.weights = {k: round(v / total, 4) for k, v in w.items()}

    # Step 6 — LOW quality fallback
    if quality_level == "LOW":
        log_warning("JD quality LOW — merging industry fallback skills.")
        domain_key = intent.domain.lower().replace(" ", "_")
        fallback = FALLBACK_SKILLS.get(domain_key, FALLBACK_SKILLS["general"])
        existing = set(s.lower() for s in intent.must_have)
        intent.must_have = intent.must_have + [
            s for s in fallback if s.lower() not in existing
        ]

    return intent


# ===========================================================================
# STREAM 2 — RESUME NORMALIZER (helpers)
# ===========================================================================

def _extract_section(text: str, header_words: list[str]) -> str:
    """
    Extract the text block that follows a named section header.

    Searches for any of the header_words as a standalone line (case-insensitive),
    then returns everything up to the next recognised section boundary.
    Returns an empty string if the header is not found.
    """
    pattern = re.compile(
        r"^(" + "|".join(re.escape(h) for h in header_words) + r")\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    next_sec = _SECTION_BOUNDARY_RE.search(text, start)
    end = next_sec.start() if next_sec else len(text)
    return text[start:end].strip()


def _extract_skills_rule_based(text: str) -> list[str]:
    """
    Extract skill tokens from resume text using two strategies:

    1. If a SKILLS section exists, parse comma- and bullet-separated items from it.
    2. Scan the full text for known tech keywords (frozenset lookup).

    Returns a deduplicated list preserving canonical display names.
    """
    found: dict[str, str] = {}  # lower → display

    # Strategy 1: explicit section
    section = _extract_section(
        text, ["skills", "technical skills", "technologies", "tech stack", "tools", "stack"]
    )
    if section:
        tokens = re.split(r"[,\n•·\-\|]+", section)
        for tok in tokens:
            tok = tok.strip()
            if 1 < len(tok) < 50:
                lower = tok.lower()
                display = _KEYWORD_DISPLAY.get(lower, tok)
                found[lower] = display

    # Strategy 2: keyword scan of full text
    text_lower = text.lower()
    for kw in _TECH_KEYWORDS:
        if kw in text_lower:
            display = _KEYWORD_DISPLAY.get(kw, kw.title())
            if kw not in found:
                found[kw] = display

    return list(found.values())


def _extract_experience_rule_based(text: str) -> list[ExperienceEntry]:
    """
    Extract work experience entries from structured resume sections.

    Looks for:
      - Pipe-delimited lines: "Title | Company | StartDate – EndDate"
      - A dedicated EXPERIENCE section with date-range patterns

    Falls back to an empty list when the text has no recognisable structure
    (e.g. paragraph-format resumes — handled by LLM Pass 2).
    """
    entries: list[ExperienceEntry] = []

    # Pattern: "Title | Company | Jan 2020 – Dec 2022"
    for m in _EXP_PIPE_RE.finditer(text):
        title, company, start_raw, end_raw = m.groups()
        start = normalize_date_string(start_raw.strip())
        end = normalize_date_string(end_raw.strip())
        entries.append(ExperienceEntry(
            title=title.strip(),
            company=company.strip(),
            start_date=start if start != "unknown" else None,
            end_date=end,
            is_current=(end == "present"),
            duration_years=_compute_duration(start, end),
        ))

    if entries:
        return entries

    # Fallback: look inside EXPERIENCE section for date ranges
    section = _extract_section(
        text, ["experience", "work experience", "employment", "work history", "career"]
    )
    if not section:
        return []

    # Find all date ranges and treat each as the boundary of a role block
    range_matches = list(_DATE_RANGE_RE.finditer(section))
    for i, rm in enumerate(range_matches):
        start_raw = rm.group(1)
        end_raw = rm.group(2)
        start = normalize_date_string(start_raw)
        end = normalize_date_string(end_raw)

        # Context before the date range: title/company usually live there
        context_start = range_matches[i - 1].end() if i > 0 else 0
        context = section[context_start:rm.start()].strip()
        lines = [l.strip() for l in context.split("\n") if l.strip()]
        title = lines[0] if lines else "Unknown Title"
        company = lines[1] if len(lines) > 1 else "Unknown Company"

        entries.append(ExperienceEntry(
            title=title,
            company=company,
            start_date=start if start != "unknown" else None,
            end_date=end,
            is_current=(end == "present"),
            duration_years=_compute_duration(start, end),
        ))

    return entries


def _extract_education_rule_based(text: str) -> list[dict]:
    """
    Extract education entries from a dedicated EDUCATION section.

    Each entry is a dict with keys: degree, institution, graduation_year.
    Returns an empty list when no education section is found.
    """
    section = _extract_section(
        text, ["education", "academic", "academic background", "qualifications"]
    )
    if not section:
        return []

    entries: list[dict] = []
    year_re = re.compile(r"\b(19|20)\d{2}\b")
    lines = [l.strip() for l in section.split("\n") if l.strip()]

    # Group every 1-2 lines into a degree entry
    i = 0
    while i < len(lines):
        line = lines[i]
        year_m = year_re.search(line)
        year = int(year_m.group()) if year_m else None

        # Pipe-split: "M.S. CS | Stanford | 2020"
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            degree = parts[0]
            institution = parts[1]
            if not year and len(parts) >= 3:
                ym = year_re.search(parts[2])
                year = int(ym.group()) if ym else None
        else:
            degree = line
            institution = lines[i + 1] if i + 1 < len(lines) else None
            if institution:
                i += 1

        entries.append({
            "degree": degree,
            "institution": institution,
            "graduation_year": year,
        })
        i += 1

    return entries if entries else []


def _extract_location(text: str) -> str | None:
    """
    Extract a location string from the first 5 lines of the resume text.

    Matches against a known set of cities (Indian + major global).
    Falls back to scanning the full text if nothing is found in the header.
    """
    lines = text.split("\n")[:5]
    header = " ".join(lines).lower()
    for city in _CITIES:
        if city in header:
            return city.title()

    # Broader scan of the whole text
    text_lower = text.lower()
    for city in _CITIES:
        if re.search(r"\b" + re.escape(city) + r"\b", text_lower):
            return city.title()

    return None


def _extract_github_url(text: str) -> str | None:
    """
    Extract the first GitHub profile URL found in the resume text.

    Returns the full https://github.com/<username> URL or None.
    """
    m = _GITHUB_RE.search(text)
    if m:
        return m.group(0)
    return None


def _compute_duration(start: str | None, end: str | None) -> float | None:
    """
    Compute the duration in years between two YYYY-MM date strings.

    Handles 'present' end dates using the current UTC date.
    Returns None when either date is missing or in an unrecognised format.
    """
    if not start or start in ("unknown", ""):
        return None

    if not end or end in ("unknown", ""):
        return None

    if end == "present":
        now = datetime.now(timezone.utc)
        end = f"{now.year}-{now.month:02d}"

    try:
        sy, sm = map(int, start.split("-"))
        ey, em = map(int, end.split("-"))
        months = (ey - sy) * 12 + (em - sm)
        return round(max(0.0, months / 12), 1)
    except (ValueError, AttributeError):
        return None


def _assess_completeness(
    skills: list[str],
    experience: list[ExperienceEntry],
    education: list[dict],
    location: str | None,
) -> tuple[str, list[str], list[str]]:
    """
    Assess Pass 1 extraction completeness and return (confidence, present, missing).

    Confidence:
      HIGH   — ≤1 field missing
      MEDIUM — 2-3 fields missing
      LOW    — >3 fields missing
    """
    present: list[str] = []
    missing: list[str] = []

    if skills:
        present.append("skills")
    else:
        missing.append("skills")

    if experience:
        present.append("experience")
    else:
        missing.append("experience")

    if education:
        present.append("education")
    else:
        missing.append("education")

    if location:
        present.append("location")
    else:
        missing.append("location")

    n_missing = len(missing)
    if n_missing <= 1:
        confidence = "HIGH"
    elif n_missing <= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return confidence, present, missing


def _run_llm_pass2(text: str, candidate_id: str) -> dict:
    """
    LLM fallback extraction for missing or unstructured resume fields.

    Caps resume text at ~6,000 characters (≈1,500 tokens) before sending.
    Returns a dict with keys: skills, experience, education, location,
    total_years_exp, name.  Returns an empty dict on failure.
    """
    capped = text[:6000]
    system_prompt = (
        "Extract structured information from this resume. "
        "Return JSON only. No markdown fences. No explanation."
    )
    user_prompt = (
        "Extract from this resume and return JSON:\n"
        "{\n"
        '  "name": "full name or null",\n'
        '  "skills": ["list of technical skills"],\n'
        '  "experience": [\n'
        '    {\n'
        '      "title": "job title",\n'
        '      "company": "company name",\n'
        '      "start_date": "YYYY-MM",\n'
        '      "end_date": "YYYY-MM or present",\n'
        '      "summary": "one sentence describing role and impact"\n'
        "    }\n"
        "  ],\n"
        '  "education": [\n'
        '    {"degree": "degree name", "institution": "school name", "graduation_year": year_int}\n'
        "  ],\n"
        '  "location": "city name or null",\n'
        '  "total_years_exp": number_or_null\n'
        "}\n\n"
        f"Resume text:\n{capped}"
    )

    raw = _call_llm(system_prompt, user_prompt, max_tokens=1024)
    result = _parse_llm_json(raw)
    if not isinstance(result, dict):
        log_warning(f"LLM Pass 2 returned non-dict for {candidate_id} — ignoring.")
        return {}
    return result


def _map_title_level(title: str) -> int:
    """
    Map a job title to a seniority level integer (1=junior … 5=executive).

    Uses regex word boundaries so "Lead" matches at end-of-string and
    "Senior" doesn't match inside unrelated words.
    Default level is 2 (mid-level IC) when no pattern matches.
    """
    t = title.lower()
    _LEVEL_RE: list[tuple[int, re.Pattern]] = [
        (1, re.compile(r'\b(intern|trainee|apprentice|student|junior|jr\.?|entry[\s\-]?level)\b')),
        (5, re.compile(r'\b(director|vp|vice[\s\-]president|cto|ceo|coo|head\s+of|chief)\b')),
        (4, re.compile(r'\b(principal|distinguished|fellow|partner)\b')),
        (4, re.compile(r'\b(lead|staff|manager|senior\s+manager|group\s+manager)\b')),
        (3, re.compile(r'\b(senior|sr\.?|specialist|consultant|iii)\b')),
    ]
    for level, pattern in _LEVEL_RE:
        if pattern.search(t):
            return level
    return 2


def _detect_fresher(
    education: list[dict],
    experience: list[ExperienceEntry],
    total_years_exp: float | None = None,
) -> bool:
    """
    Return True when the candidate is a recent graduate with ≤1 year of experience.

    Conditions (all must hold):
      - Graduation year >= current_year - 2
      - total_years_exp (from structured JSON or computed) <= 1.0
      - No full-time (non-intern/trainee) experience entries
    """
    current_year = datetime.now(timezone.utc).year
    # If structured total_years_exp says > 1, definitely not a fresher
    if total_years_exp is not None and total_years_exp > 1:
        return False
    for edu in education:
        grad_year = edu.get("graduation_year")
        if not grad_year:
            continue
        try:
            if current_year - int(grad_year) <= 2:
                non_intern = [
                    e for e in experience
                    if e.title and not any(
                        kw in e.title.lower()
                        for kw in ("intern", "student", "trainee", "assistant", "research assistant")
                    )
                ]
                if not non_intern:
                    return True
        except (TypeError, ValueError):
            continue
    return False


def _detect_career_changer(
    experience: list[ExperienceEntry],
    education: list[dict],
) -> bool:
    """
    Return True when the candidate has a significant career discontinuity.

    Triggers:
      - MBA degree in education
      - Founder / co-founder title in experience
      - 24+ month gap between consecutive experience entries
    """
    # MBA education
    for edu in education:
        deg = (edu.get("degree") or "").lower()
        if "mba" in deg or "m.b.a" in deg:
            return True

    # Founder title
    for exp in experience:
        title = (exp.title or "").lower()
        if any(kw in title for kw in ("founder", "co-founder", "cofounder")):
            return True

    # 2+ year gap between consecutive dated roles
    dated = [
        e for e in experience
        if e.start_date and e.start_date not in ("unknown", "")
    ]
    dated.sort(key=lambda e: e.start_date or "")
    for i in range(1, len(dated)):
        prev_end = dated[i - 1].end_date
        curr_start = dated[i].start_date
        if not prev_end or prev_end in ("present", "unknown", ""):
            continue
        try:
            py, pm = map(int, prev_end.split("-"))
            cy, cm = map(int, curr_start.split("-"))
            gap_months = (cy - py) * 12 + (cm - pm)
            if gap_months >= 24:
                return True
        except (ValueError, AttributeError):
            continue
    return False


def _compute_total_years(experience: list[ExperienceEntry]) -> float | None:
    """
    Sum duration_years across all experience entries.

    Returns None (not 0) when no dated entries are available, so downstream
    signals know there is genuinely no data rather than zero experience.
    """
    durations = [e.duration_years for e in experience if e.duration_years is not None]
    if not durations:
        return None
    return round(sum(durations), 1)


def _compute_trajectory(experience: list[ExperienceEntry]) -> str:
    """
    Derive a career trajectory label from the sequence of job titles.

    Labels (in evaluation order):
      startup_gap        — gap >= 12 months immediately before founder role
      consulting_pattern — 3+ senior roles at 3+ different companies
      high_growth        — average level gain > 0.5 levels/year
      steady_climb       — 0.2–0.5 levels/year
      lateral            — 0.0–0.2 levels/year
      stagnant           — same level for 5+ years
      declining          — average level loss per year
      insufficient_data  — < 2 roles or no dated entries
    """
    dated = [
        e for e in experience
        if e.start_date and e.start_date not in ("unknown", "")
    ]
    if len(dated) < 2:
        return "insufficient_data"

    dated.sort(key=lambda e: e.start_date or "")

    # Exception: startup gap — gap + founder title
    for i in range(1, len(dated)):
        prev_end = dated[i - 1].end_date
        curr_start = dated[i].start_date
        if prev_end and prev_end not in ("present", "unknown", "") and curr_start:
            try:
                py, pm = map(int, prev_end.split("-"))
                cy, cm = map(int, curr_start.split("-"))
                gap_months = (cy - py) * 12 + (cm - pm)
                if gap_months >= 12 and any(
                    kw in (dated[i].title or "").lower()
                    for kw in ("founder", "co-founder", "ceo", "cto")
                ):
                    return "startup_gap"
            except (ValueError, AttributeError):
                pass

    # Exception: consulting pattern
    senior_entries = [e for e in dated if _map_title_level(e.title or "") >= 3]
    unique_companies = {e.company for e in senior_entries if e.company}
    if len(senior_entries) >= 3 and len(unique_companies) >= 3:
        return "consulting_pattern"

    # Compute level trajectory over time
    levels = [_map_title_level(e.title or "") for e in dated]
    first_start = dated[0].start_date or ""
    last_end = dated[-1].end_date
    if last_end == "present" or not last_end:
        now = datetime.now(timezone.utc)
        last_end = f"{now.year}-{now.month:02d}"

    try:
        sy, sm = map(int, first_start.split("-"))
        ey, em = map(int, last_end.split("-"))
        total_years = max(1.0, ((ey - sy) * 12 + (em - sm)) / 12)
    except (ValueError, AttributeError):
        total_years = max(1.0, float(len(dated)))

    level_change_per_year = (levels[-1] - levels[0]) / total_years

    # Stagnant: same level, 5+ years
    if len(set(levels)) == 1 and total_years >= 5:
        return "stagnant"

    if level_change_per_year > 0.5:
        return "high_growth"
    if level_change_per_year >= 0.2:
        return "steady_climb"
    if level_change_per_year >= 0.0:
        return "lateral"
    return "declining"


# ===========================================================================
# STREAM 2 — RESUME NORMALIZER (main function)
# ===========================================================================

def normalize_resume(
    raw: CandidateRaw,
    pre_populated: dict | None = None,
) -> CandidateNormalized:
    """
    Normalize a single raw resume into a CandidateNormalized record.

    pre_populated: optional full candidate dict from structured JSON.
      Structured fields (skills, experience, education, github_url,
      total_years_exp) are loaded first; Pass 1 extends them from text;
      Pass 2 LLM only fires if critical fields are still missing.

    Pass 1: rule-based regex extraction (supplements pre_populated).
    Pass 2: GPT-4o-mini fallback when confidence is LOW or critical
            fields (skills / experience) remain missing after both
            pre_populated and Pass 1.
    """
    text = raw.raw_text or ""
    words = text.split()
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    non_ascii_ratio = non_ascii / max(len(text), 1)

    # Scan-PDF guard — only bail out when text is genuinely unreadable garbage:
    # the heuristic must fire AND the text must be very short AND mostly non-ASCII.
    # Short-but-readable synthetic/incomplete profiles must NOT be caught here.
    if detect_scan_pdf(text) and len(words) < 15 and non_ascii_ratio > 0.4:
        log_warning(f"{raw.candidate_id}: scan-only PDF detected — LOW confidence, skipping LLM.")
        return CandidateNormalized(
            candidate_id=raw.candidate_id,
            raw_text=text,
            confidence_level="LOW",
            trajectory_label="insufficient_data",
            fields_missing=["skills", "experience", "education", "location"],
        )

    # Very short but readable text: flag as incomplete, continue normalization.
    if len(words) < 20:
        log_warning(f"{raw.candidate_id}: very short profile — LOW confidence")

    # --- Pre-populate from structured JSON (when provided) ---
    pre_total_years: float | None = None
    if pre_populated:
        # Skills: seed list; Pass 1 will supplement but never replace
        p_pre_skills: list[str] = [str(s) for s in (pre_populated.get("skills") or []) if s]

        # Experience: convert JSON dicts → ExperienceEntry objects
        p_pre_experience: list[ExperienceEntry] = []
        for entry in (pre_populated.get("experience") or []):
            if not isinstance(entry, dict):
                continue
            start = normalize_date_string(str(entry.get("start_date") or ""))
            end = normalize_date_string(str(entry.get("end_date") or ""))
            p_pre_experience.append(ExperienceEntry(
                title=entry.get("title") or "Unknown",
                company=entry.get("company") or "Unknown",
                start_date=start if start != "unknown" else None,
                end_date=end,
                summary=entry.get("summary"),
                is_current=(end == "present"),
                duration_years=_compute_duration(start, end),
            ))

        # Education: normalize to list[dict]
        raw_edu = pre_populated.get("education")
        if isinstance(raw_edu, dict):
            p_pre_education: list[dict] = [raw_edu]
        elif isinstance(raw_edu, list):
            p_pre_education = [e for e in raw_edu if isinstance(e, dict)]
        else:
            p_pre_education = []

        pre_github: str | None = pre_populated.get("github_url") or None
        pre_name: str | None = pre_populated.get("name") or None
        try:
            pre_total_years = round(float(pre_populated.get("total_years_exp") or 0), 1) or None
        except (TypeError, ValueError):
            pre_total_years = None
    else:
        p_pre_skills, p_pre_experience, p_pre_education, pre_github, pre_name = [], [], [], None, None

    # --- Pass 1: rule-based (extends pre-populated, never replaces) ---
    p1_skills_text = _extract_skills_rule_based(text)
    existing_lower = {s.lower() for s in p_pre_skills}
    p1_skills = p_pre_skills + [s for s in p1_skills_text if s.lower() not in existing_lower]

    # Use pre-populated structured lists when available; extract from text otherwise
    p1_experience = p_pre_experience if p_pre_experience else _extract_experience_rule_based(text)
    p1_education = p_pre_education if p_pre_education else _extract_education_rule_based(text)

    p1_location = _extract_location(text)
    github_url = pre_github or _extract_github_url(text)

    confidence, fields_present, fields_missing = _assess_completeness(
        p1_skills, p1_experience, p1_education, p1_location
    )

    # --- Pass 2: LLM fallback ---
    fields_inferred: list[str] = []
    name: str | None = pre_name  # seed from JSON; LLM can overwrite if it finds better
    p2_data: dict = {}

    needs_pass2 = (
        confidence == "LOW"
        or "skills" in fields_missing
        or "experience" in fields_missing
    )

    if needs_pass2:
        p2_data = _run_llm_pass2(text, raw.candidate_id)

        # Merge — Pass 1 takes priority; LLM supplements missing skills
        if p2_data.get("skills"):
            llm_skills = [str(s) for s in p2_data["skills"] if s]
            existing_lower = {s.lower() for s in p1_skills}
            added = [s for s in llm_skills if s.lower() not in existing_lower]
            if added:
                p1_skills = p1_skills + added
                if "skills" in fields_missing:
                    fields_missing.remove("skills")
                    fields_present.append("skills")
                if "skills" not in fields_inferred:
                    fields_inferred.append("skills")

        if not p1_experience and p2_data.get("experience"):
            for entry in p2_data["experience"]:
                if not isinstance(entry, dict):
                    continue
                start = normalize_date_string(str(entry.get("start_date") or ""))
                end = normalize_date_string(str(entry.get("end_date") or ""))
                p1_experience.append(ExperienceEntry(
                    title=entry.get("title") or "Unknown",
                    company=entry.get("company") or "Unknown",
                    start_date=start if start != "unknown" else None,
                    end_date=end,
                    summary=entry.get("summary"),
                    is_current=(end == "present"),
                    duration_years=_compute_duration(start, end),
                ))
            if p1_experience:
                if "experience" in fields_missing:
                    fields_missing.remove("experience")
                    fields_present.append("experience")
                fields_inferred.append("experience")

        if not p1_education and p2_data.get("education"):
            raw_edu = p2_data["education"]
            if isinstance(raw_edu, dict):
                raw_edu = [raw_edu]
            for edu in raw_edu:
                if isinstance(edu, dict):
                    p1_education.append(edu)
            if p1_education:
                if "education" in fields_missing:
                    fields_missing.remove("education")
                    fields_present.append("education")
                fields_inferred.append("education")

        if not p1_location and p2_data.get("location"):
            p1_location = str(p2_data["location"])
            if "location" in fields_missing:
                fields_missing.remove("location")
                fields_present.append("location")
            fields_inferred.append("location")

        name = p2_data.get("name") or None

        # Re-assess confidence after merge
        confidence, fields_present, fields_missing = _assess_completeness(
            p1_skills, p1_experience, p1_education, p1_location
        )

    # Total years: computed from experience entries, then fallback to pre-populated, then LLM
    total_years = _compute_total_years(p1_experience)
    if total_years is None and pre_total_years is not None:
        total_years = pre_total_years
    if total_years is None and p2_data.get("total_years_exp") is not None:
        try:
            total_years = round(float(p2_data["total_years_exp"]), 1)
        except (TypeError, ValueError):
            pass

    # Derived fields
    trajectory = _compute_trajectory(p1_experience)
    if trajectory is None:          # guaranteed guard — _compute_trajectory should never return None
        trajectory = "insufficient_data"
    is_fresher = _detect_fresher(p1_education, p1_experience, total_years_exp=total_years)
    is_career_changer = _detect_career_changer(p1_experience, p1_education)

    return CandidateNormalized(
        candidate_id=raw.candidate_id,
        raw_text=text,
        name=name,
        skills=p1_skills,
        experience=p1_experience,
        education=p1_education if p1_education else None,
        location=p1_location,
        total_years_exp=total_years,
        trajectory_label=trajectory,
        github_url=github_url,
        confidence_level=confidence,
        language_detected="en",
        is_fresher=is_fresher,
        is_career_changer=is_career_changer,
        fields_present=fields_present,
        fields_inferred=fields_inferred,
        fields_missing=fields_missing,
    )


# ===========================================================================
# DEDUP FILTER
# ===========================================================================

def _get_institution(education: list[dict] | dict | None) -> str:
    """Extract the institution name from an education field (list or dict)."""
    if isinstance(education, dict):
        return (education.get("institution") or "").lower().strip()
    if isinstance(education, list) and education:
        first = education[0]
        if isinstance(first, dict):
            return (first.get("institution") or "").lower().strip()
    return ""


def dedup_candidates(
    candidates: list[CandidateNormalized],
    embeddings: np.ndarray,
) -> list[CandidateNormalized]:
    """
    Remove duplicate candidate records using two complementary checks.

    PRIMARY (embedding cosine similarity > 0.92):
      Near-identical resumes submitted under different IDs are caught here.

    SECONDARY (metadata):
      For pairs that pass the embedding threshold, check:
        same total_years_exp (±0.5 yr)
        AND top-3 skills overlap >= 2
        AND same institution

    Resolution: keep the record with higher confidence_level.
    If confidence is equal, keep the first occurrence.
    Flagged duplicates are logged and excluded from the return list.
    """
    if len(candidates) <= 1:
        return candidates

    _CONF_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}

    # Normalise embeddings → cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms
    sim_matrix: np.ndarray = normed @ normed.T

    is_dup = [False] * len(candidates)

    for i in range(len(candidates)):
        if is_dup[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if is_dup[j]:
                continue

            flagged = False

            # Primary: embedding similarity
            if sim_matrix[i, j] > 0.92:
                flagged = True
            else:
                # Secondary: metadata check
                ci, cj = candidates[i], candidates[j]
                skills_i = {s.lower() for s in (ci.skills or [])[:3]}
                skills_j = {s.lower() for s in (cj.skills or [])[:3]}
                skills_overlap = len(skills_i & skills_j)

                years_i = ci.total_years_exp
                years_j = cj.total_years_exp
                exp_diff = abs((years_i or 0) - (years_j or 0))
                same_years = (
                    years_i is not None
                    and years_j is not None
                    and exp_diff <= 0.3  # tightened from 0.5 — avoids false positives
                )

                inst_i = _get_institution(ci.education)
                inst_j = _get_institution(cj.education)
                same_inst = bool(inst_i and inst_j and inst_i == inst_j)

                name_i = (ci.name or "").lower()
                name_j = (cj.name or "").lower()
                name_sim = (
                    difflib.SequenceMatcher(None, name_i, name_j).ratio()
                    if name_i and name_j else 0.0
                )

                # Path A: full metadata match — skills + institution + years + name
                path_a = (
                    same_years
                    and skills_overlap >= 2
                    and same_inst
                    and name_sim >= 0.6
                )

                # Path B: strong identity match — high name similarity + institution +
                # years. Catches incomplete duplicate profiles that have < 2 skills listed.
                path_b = name_sim >= 0.80 and same_inst and same_years

                if path_a or path_b:
                    reason = "path_b:strong-identity" if (path_b and not path_a) else "path_a:full-metadata"
                    log_warning(
                        f"Dedup secondary ({reason}): {ci.candidate_id} exp={years_i} vs "
                        f"{cj.candidate_id} exp={years_j} diff={exp_diff:.3f} "
                        f"name_sim={name_sim:.2f} skills_overlap={skills_overlap}"
                    )
                    flagged = True

            if flagged:
                ci_rank = _CONF_RANK.get(candidates[i].confidence_level, 0)
                cj_rank = _CONF_RANK.get(candidates[j].confidence_level, 0)
                dup_idx = j if ci_rank >= cj_rank else i
                is_dup[dup_idx] = True
                log_warning(
                    f"Duplicate: {candidates[i].candidate_id} ≈ "
                    f"{candidates[j].candidate_id} "
                    f"(sim={sim_matrix[i, j]:.2f}) — "
                    f"dropping {candidates[dup_idx].candidate_id}"
                )

    return [c for idx, c in enumerate(candidates) if not is_dup[idx]]


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def run_layer0(
    jd_text: str,
    job_id: str,
    candidates_raw: list[dict],
) -> tuple[JDIntent, list[CandidateNormalized]]:
    """
    Top-level Layer 0 orchestrator.

    candidates_raw: full candidate dicts from candidates.json.  Each dict
      is passed to normalize_resume as pre_populated so structured fields
      (skills, experience, education, github_url, total_years_exp) flow
      through without requiring LLM extraction.

    Steps:
      1. Log pipeline start.
      2. Build CandidateRaw objects for input validation.
      3. Deduplicate raw candidate IDs.
      4. Stream 1: parse_jd → JDIntent.
      5. Stream 2: normalize_resume for each candidate (tqdm progress bar).
      6. Assert at least one candidate has a non-trivial trajectory.
      7. Embed all normalized candidates with all-MiniLM-L6-v2 for dedup.
      8. dedup_candidates → remove near-identical submissions.
      9. Log layer complete.
     10. Return (JDIntent, normalized_candidates).
    """
    import time as _time
    t0 = _time.time()

    run_id = generate_run_id()
    log_pipeline_start(run_id, f"Layer 0 — {job_id}", len(candidates_raw))

    # Step 2 — build CandidateRaw objects for validation, then validate
    raw_objects: list[CandidateRaw] = [
        CandidateRaw(
            candidate_id=str(c.get("candidate_id", "")),
            raw_text=str(c.get("raw_text", "")),
        )
        for c in candidates_raw
    ]
    ok, errors = validate_pipeline_inputs(jd_text, raw_objects)
    if not ok:
        raise ValueError("Pipeline input validation failed:\n" + "\n".join(errors))

    # Step 3 — ID dedup (same ID, different text)
    raw_objects = ensure_unique_candidate_ids(raw_objects)
    valid_ids = {r.candidate_id for r in raw_objects}

    # Rebuild dict list in the de-duped order; map by ID for O(1) lookup
    id_to_dict: dict[str, dict] = {str(c.get("candidate_id", "")): c for c in candidates_raw}

    # Step 4 — JD parsing
    print("  → Parsing JD...")
    jd_intent = parse_jd(jd_text, job_id)
    print(
        f"  → JD parsed: title={jd_intent.job_title!r} | "
        f"quality={jd_intent.quality_score:.0f} ({jd_intent.quality_level}) | "
        f"must_have={len(jd_intent.must_have)} skills"
    )

    # Step 5 — resume normalization (pre_populated carries full structured JSON)
    normalized: list[CandidateNormalized] = []
    for raw in tqdm(raw_objects, desc="  Normalizing resumes", unit="resume"):
        full_dict = id_to_dict.get(raw.candidate_id, {})
        norm = normalize_resume(raw, pre_populated=full_dict)
        normalized.append(norm)

    # Step 6 — sanity check: structured fields must be flowing through
    non_insufficient = [c for c in normalized if c.trajectory_label != "insufficient_data"]
    assert len(non_insufficient) > 0, (
        "All trajectories are insufficient_data — structured fields not flowing through"
    )

    # Step 7 — embed for dedup
    print("  → Loading sentence-transformer for dedup embeddings...")
    model = SentenceTransformer(_EMBEDDING_MODEL)
    texts = [n.raw_text for n in normalized]
    embeddings: np.ndarray = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    )

    # Step 8 — content dedup
    before_dedup = len(normalized)
    normalized = dedup_candidates(normalized, embeddings)
    removed = before_dedup - len(normalized)
    if removed:
        print(f"  → Dedup removed {removed} duplicate(s).")

    elapsed = _time.time() - t0
    log_layer_complete(0, len(normalized), elapsed)

    return jd_intent, normalized


# ===========================================================================
# QUICK TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    import dataclasses

    DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

    # --- Load JD ---
    jd_path = DATA_DIR / "jd_senior_ml.txt"
    jd_raw = json.loads(jd_path.read_text(encoding="utf-8"))
    # Convert structured JSON to flat text the LLM can parse
    jd_obj = jd_raw.get("job_description", jd_raw)
    jd_parts: list[str] = []
    for key in ("role_overview", "responsibilities", "must_have_requirements",
                "nice_to_have", "preferred_qualifications", "technical_requirements",
                "compensation_and_benefits", "about_the_role"):
        val = jd_obj.get(key)
        if isinstance(val, str):
            jd_parts.append(val)
        elif isinstance(val, list):
            jd_parts.extend(str(v) for v in val)
    jd_text = "\n".join(jd_parts) or json.dumps(jd_obj)

    # --- Load all 30 candidates (pass full dicts — pre_populated uses them) ---
    all_candidates = json.loads(
        (DATA_DIR / "candidates.json").read_text(encoding="utf-8")
    )
    candidates_raw = all_candidates  # full 30 so trajectory assert passes (multi-role candidates exist)

    print("\n" + "=" * 55)
    print("  Layer 0 Quick Test — jd_senior_ml.txt + 30 candidates")
    print("=" * 55 + "\n")

    jd_intent, normalized = run_layer0(jd_text, "JD_SENIOR_ML", candidates_raw)

    # --- Print JDIntent ---
    print("\n── JD Intent ──────────────────────────────────────────")
    print(json.dumps(dataclasses.asdict(jd_intent), indent=2))

    # --- Print candidate summaries ---
    print("\n── Candidate Summaries (first 3) ───────────────────────")
    for c in normalized[:3]:
        print(
            f"  {c.candidate_id} | conf={c.confidence_level:6s} | "
            f"traj={c.trajectory_label or 'n/a':22s} | "
            f"fresher={c.is_fresher} | changer={c.is_career_changer}"
        )
        print(f"    present : {c.fields_present}")
        print(f"    inferred: {c.fields_inferred}")
        print(f"    missing : {c.fields_missing}")
        print(f"    skills  : {c.skills[:5]}")
        print(f"    exp_yrs : {c.total_years_exp}")
        print()

    # --- Print dedup summary ---
    print(f"── Dedup: {len(candidates_raw)} input → {len(normalized)} output ─────────")
    for c in normalized:
        print(f"  {c.candidate_id}")
    print()
