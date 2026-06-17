"""
generate_dataset.py — Synthetic candidate dataset generator for HireSynth.

Calls GPT-4o-mini once per candidate group to produce 30 realistic profiles
targeting a Senior ML Engineer role at an Indian product startup. Also
generates three job-description files spanning HIGH / MEDIUM / LOW quality.

Outputs:
    data/synthetic/candidates.json       — 30 candidate profiles
    data/synthetic/metadata.json         — coverage summary
    data/synthetic/jd_senior_ml.txt      — HIGH quality JD
    data/synthetic/jd_data_scientist.txt — MEDIUM quality JD
    data/synthetic/jd_mlops_vague.txt    — LOW quality JD (tests fallback)

Usage:
    python generate_dataset.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env", override=True)

import json
import os
import re
import time
from datetime import datetime, timezone
import numpy as np
from openai import OpenAI

np.random.seed(42)

# ---------------------------------------------------------------------------
# API key validation — fail fast with a clear message
# ---------------------------------------------------------------------------

_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key or _api_key == "sk-your-key-here" or len(_api_key) < 20:
    raise ValueError(
        "Valid OPENAI_API_KEY not found in .env file. "
        "Check /Users/praneethv/Desktop/hiresynth/.env"
    )

client = OpenAI(api_key=_api_key)

# ---------------------------------------------------------------------------
# Paths and group registry
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/synthetic")
CANDIDATES_FILE = OUTPUT_DIR / "candidates.json"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# (type_label, count, first_candidate_id_int)
GROUP_SPECS: list[tuple[str, int, int]] = [
    ("strong_match",       5,  1),
    ("good_match",         5,  6),
    ("hidden_gem",         3, 11),
    ("fresher",            2, 14),
    ("overqualified",      2, 16),
    ("career_changer",     2, 18),
    ("incomplete_profile", 3, 20),
    ("duplicate",          2, 23),
    ("github_active",      2, 25),
    ("poor_match",         4, 27),
]

_INDIAN_STARTUPS = (
    "Flipkart, Swiggy, Zomato, CRED, PhonePe, Meesho, Razorpay, Zepto, "
    "Ola, Dream11, Freshworks, Zoho, InMobi, Urban Company, Groww, Lenskart, "
    "Google India (Bangalore), Microsoft India, Amazon India, Samsung R&D India"
)
_INDIAN_SCHOOLS = (
    "IIT Bombay, IIT Madras, IIT Delhi, IIT Kharagpur, IISc Bangalore, "
    "BITS Pilani, IIIT Hyderabad, NIT Trichy, NIT Surathkal, VIT Vellore"
)

# Exact JSON schema the LLM must follow
_SCHEMA_RULES = """
Return a raw JSON array — no markdown fences, no prose. Each element:
{
  "candidate_id": "C001",
  "type": "<group_type_label>",
  "name": "Full Name",
  "skills": ["Skill1", "Skill2"],
  "experience": [
    {
      "title": "Job Title",
      "company": "Company Name",
      "start_date": "2021-03",
      "end_date": "2024-08",
      "summary": "One or two sentences about what they did and the impact."
    }
  ],
  "education": {
    "degree": "M.Tech Computer Science",
    "institution": "IIT Bombay",
    "graduation_year": 2019
  },
  "location": "Bangalore",
  "total_years_exp": 5.0,
  "github_url": "https://github.com/username",
  "raw_text": "Single flowing paragraph summarising the resume.",
  "confidence_level": "HIGH"
}

Hard rules:
• education is a SINGLE object, NOT a list
• start_date / end_date are "YYYY-MM" strings; current role has end_date "present"
• github_url is a full https:// URL or JSON null
• raw_text is one plain-text paragraph (no bullet points)
• Output ONLY the JSON array — nothing else
"""


# ---------------------------------------------------------------------------
# 1. LLM wrapper
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 4096,
    retries: int = 3,
) -> str:
    """
    Call GPT-4o-mini with the given prompt and return the raw text response.

    Retries up to `retries` times with exponential back-off on transient
    failures. Raises on the final failure.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise JSON data generator for HR/recruitment "
                            "pipelines. Return only valid JSON — no markdown, no prose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    ⚠ LLM error (attempt {attempt + 1}/{retries}): {exc} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise
    return ""


# ---------------------------------------------------------------------------
# 3. JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> list[dict]:
    """
    Strip markdown fences (if present) and parse a JSON array from LLM output.

    Returns an empty list if parsing fails; caller is responsible for retrying.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError as exc:
        print(f"    ⚠ JSON parse error: {exc}")
        print(f"    Response preview: {text[:200]}")
    return []


# ---------------------------------------------------------------------------
# 4. Candidate validation and normalisation
# ---------------------------------------------------------------------------

def validate_candidate(raw: dict, candidate_id: str, group_type: str) -> dict:
    """
    Normalise one raw LLM-generated candidate dict to the locked schema.

    - Enforces the correct candidate_id and type.
    - Coerces education from list → dict if the LLM returned a list.
    - Renames experience.description → experience.summary for schema consistency.
    - Fills missing fields with safe defaults.
    """
    # education: must be a single dict
    education = raw.get("education")
    if isinstance(education, list):
        education = education[0] if education else {}
    if not isinstance(education, dict):
        education = {"degree": None, "institution": None, "graduation_year": None}

    # experience: normalise description → summary
    raw_exp = raw.get("experience") or []
    experience: list[dict] = []
    for entry in raw_exp:
        if not isinstance(entry, dict):
            continue
        if "description" in entry and "summary" not in entry:
            entry["summary"] = entry.pop("description")
        experience.append(entry)

    return {
        "candidate_id": candidate_id,
        "type": group_type,
        "name": str(raw.get("name") or "Unknown"),
        "skills": list(raw.get("skills") or []),
        "experience": experience,
        "education": education,
        "location": raw.get("location"),
        "total_years_exp": float(raw.get("total_years_exp") or 0.0),
        "github_url": raw.get("github_url"),
        "raw_text": str(raw.get("raw_text") or ""),
        "confidence_level": str(raw.get("confidence_level") or "LOW"),
    }


# ---------------------------------------------------------------------------
# 5. Post-processing fixups for structurally-special groups
# ---------------------------------------------------------------------------

def _fixup_incomplete_profile(candidates: list[dict]) -> None:
    """
    Enforce the exact structural gaps required for the incomplete_profile group.

    C020 (index 0): location → null, first experience entry loses its dates.
    C021 (index 1): skills → empty list, confidence → LOW.
    C022 (index 2): all structured fields nulled; only raw_text + id + type kept.
    """
    if len(candidates) < 1:
        return

    # C020 — missing location + missing dates on first exp entry
    c = candidates[0]
    c["location"] = None
    c["confidence_level"] = "MEDIUM"
    if c["experience"]:
        c["experience"][0]["start_date"] = None
        c["experience"][0]["end_date"] = None

    if len(candidates) < 2:
        return

    # C021 — skills list stripped (skills live only in raw_text)
    c = candidates[1]
    c["skills"] = []
    c["confidence_level"] = "LOW"

    if len(candidates) < 3:
        return

    # C022 — skeleton record; only raw_text carries any information
    c = candidates[2]
    c["skills"] = []
    c["experience"] = []
    c["education"] = {"degree": None, "institution": None, "graduation_year": None}
    c["location"] = None
    c["github_url"] = None
    c["confidence_level"] = "LOW"
    c["total_years_exp"] = 0.0


def _fixup_duplicate(candidates: list[dict]) -> None:
    """
    Ensure the duplicate pair shares identical companies, dates, and github_url.

    C023 is the reference resume; C024 is the LinkedIn-export twin.
    """
    if len(candidates) < 2:
        return
    c1, c2 = candidates[0], candidates[1]

    # Overwrite C024 experience with C023 experience (same dates, same companies)
    c2["experience"] = [dict(e) for e in c1["experience"]]
    c2["github_url"] = c1["github_url"]
    c2["total_years_exp"] = c1["total_years_exp"]

    # Drop one skill to make C024 slightly different
    if len(c2["skills"]) > 1:
        c2["skills"] = c2["skills"][:-1]


def _fixup_github_active(candidates: list[dict]) -> None:
    """
    Trim skills to a short generic list so resume text looks thin.

    GitHub URL must still be present so the enrichment layer can boost them.
    """
    generic_skills = ["Python", "SQL", "Git"]
    for c in candidates:
        c["skills"] = generic_skills[:]
        c["confidence_level"] = "MEDIUM"
        # Truncate raw_text to feel vague and short
        words = c["raw_text"].split()
        if len(words) > 40:
            c["raw_text"] = " ".join(words[:40]) + "."


# ---------------------------------------------------------------------------
# 6. Group prompt builder
# ---------------------------------------------------------------------------

def build_group_prompt(group_type: str, count: int, ids: list[str]) -> str:
    """
    Return the GPT-4o-mini prompt for one candidate group.

    Uses if/elif branches (not a pre-built dict) so only the selected branch
    is evaluated — no eager indexing into ids for the wrong group.
    Candidate slots that vary per-position are built with zip(ids, ...) so
    no hardcoded ids[n] can ever raise an IndexError.
    """
    assert len(ids) == count, (
        f"ID count mismatch: {len(ids)} ids for {count} candidates "
        f"(group_type={group_type!r})"
    )

    id_str = ", ".join(ids)
    target_jd = (
        "Target Job: Senior ML Engineer at an Indian product startup.\n"
        "Must-have: Python, PyTorch or TensorFlow, ML model training, 3+ years exp.\n"
        "Nice-to-have: MLOps, Kubernetes, distributed training.\n"
        "Seniority: senior (5–8 years ideal). Location: Chennai or Bangalore (or remote).\n"
    )

    if group_type == "strong_match":
        body = f"""
Generate {count} Senior ML Engineers who are strong matches for the target JD.

Candidate IDs (use in order): {id_str}
type field: "strong_match"

Requirements:
- 5–8 years of experience in ML engineering roles
- Skills: Python + (PyTorch or TensorFlow) + at least 2 of [MLOps, Kubernetes, distributed training, MLflow]
- Experience at well-known Indian product companies: {_INDIAN_STARTUPS}
- Active GitHub URL: https://github.com/<realistic_lowercase_username>
- Locations: Bangalore or Chennai ONLY
- Realistic Indian names (mix of South and North Indian first names, common Indian surnames)
- confidence_level: "HIGH"
- Each of the {count} candidates must be meaningfully distinct (different skill emphasis, different company)
"""

    elif group_type == "good_match":
        body = f"""
Generate {count} ML engineers who fit the JD well but are missing 1–2 nice-to-haves.

Candidate IDs (use in order): {id_str}
type field: "good_match"

Requirements:
- 3–5 years of experience
- Has Python + one ML framework (PyTorch or TF); MISSING 1–2 of [Kubernetes, MLOps, distributed training]
- Each candidate must have a DIFFERENT gap (one missing K8s, another missing MLOps, etc.)
- Experience at Indian tech companies or IT services firms: {_INDIAN_STARTUPS}
- Locations: mix — 3 in Bangalore/Chennai, 2 remote
- 2 of the {count} should have no GitHub URL (null)
- confidence_level: "HIGH"
"""

    elif group_type == "hidden_gem":
        body = f"""
Generate {count} candidates with high-growth trajectory but 60–75% skill match.

Candidate IDs (use in order): {id_str}
type field: "hidden_gem"

Requirements:
- 4–5 years exp showing RAPID growth (e.g., promoted twice, lateral upskilling, analyst→MLE)
- Skills: Python + ONE ML framework only — missing cloud infra, Kubernetes, MLOps tooling
- Strong active GitHub (ML project repos, recent commits) — github_url must be present
- raw_text should hint at high-growth trajectory even though structured skills are sparse
- Locations: Bangalore, Chennai, or Hyderabad
- These candidates LOOK underrated on paper but their growth arc shows high potential
- confidence_level: "HIGH"
"""

    elif group_type == "fresher":
        body = f"""
Generate {count} recent Indian engineering graduates applying for ML roles.

Candidate IDs (use in order): {id_str}
type field: "fresher"

Requirements:
- Graduated 2023 or 2024 from an Indian engineering college: {_INDIAN_SCHOOLS}
- 0–1 years total experience — experience entries must be INTERNSHIPS, research assistantships,
  or capstone projects ONLY (no full-time employment)
- Skills learned via coursework, Kaggle, personal projects, MOOCs (Coursera, deeplearning.ai)
- GitHub present with student project repos (e.g., https://github.com/<name>-ml-projects)
- Locations: Bangalore or Chennai (recently relocated or about to)
- raw_text should read like a student resume — enthusiastic, references college projects
- confidence_level: "HIGH"
"""

    elif group_type == "overqualified":
        body = f"""
Generate {count} very senior candidates who are over-qualified for a Senior ML Engineer IC role.

Candidate IDs (use in order): {id_str}
type field: "overqualified"

Requirements:
- 12–18 years of experience
- Current or recent titles: VP Engineering, Principal ML Architect, CTO, Director of AI
- Strong technical ML skills but responsibilities are now org-level (strategy, budgets, large teams)
- Makes it obvious they would be applying for a role BELOW their level
- Experience at large Indian/global companies (TCS, Infosys at exec level; or MNCs)
- github_url: null (executives rarely maintain active public GitHub)
- confidence_level: "HIGH"
- Include team sizes managed, budget owned, org-level decisions
"""

    elif group_type == "career_changer":
        _cc_specs = [
            (
                "Startup founder gap",
                "- 2-year gap from 2020-01 to 2022-03 during which they CO-FOUNDED a startup "
                "(which failed or was acqui-hired)\n"
                "- raw_text and experience summary must use the word \"co-founded\"\n"
                "- Before the gap: 3+ years as ML/data engineer at a reputable company\n"
                "- After the gap (2022-present): returned to an ML role\n"
                "- Skills present but slightly dated\n"
                "- confidence_level: \"MEDIUM\"\n"
                "- Location: Bangalore",
            ),
            (
                "MBA career changer",
                "- Did a full-time MBA at an Indian business school "
                "(IIM Ahmedabad, IIM Bangalore, or ISB Hyderabad)\n"
                "- Before MBA: 2–3 years in software or data analyst roles\n"
                "- During MBA (2 year gap): no coding, full-time student\n"
                "- After MBA: completed Coursera ML specialisation + deeplearning.ai courses "
                "to pivot back to ML\n"
                "- raw_text mentions specific online certifications by name\n"
                "- confidence_level: \"MEDIUM\"\n"
                "- Location: Chennai or Bangalore",
            ),
        ]
        slots = "\n\n".join(
            f"Candidate {i + 1} (ID {cid}) — {label}:\n{desc}"
            for i, (cid, (label, desc)) in enumerate(zip(ids, _cc_specs))
        )
        body = f"""
Generate exactly {count} career-changer candidates.

Candidate IDs (use in order): {id_str}
type field: "career_changer"

{slots}
"""

    elif group_type == "incomplete_profile":
        _ip_specs = [
            (
                "Missing location + missing dates",
                "- Real ML skills and experience, but set location to null\n"
                "- One experience entry must have start_date null and end_date null (dates unknown)\n"
                "- All other fields filled reasonably\n"
                "- confidence_level: \"MEDIUM\"",
            ),
            (
                "Empty skills array",
                "- Set skills to an EMPTY ARRAY: []\n"
                "- Skills ARE described in raw_text paragraph but NOT in the skills array field\n"
                "- Otherwise has reasonable experience and education\n"
                "- confidence_level: \"LOW\"",
            ),
            (
                "Raw text only",
                "- Set skills: [], experience: [], location: null, github_url: null\n"
                "- education: {\"degree\": null, \"institution\": null, \"graduation_year\": null}\n"
                "- Only raw_text contains any information (make it an ML-relevant paragraph)\n"
                "- confidence_level: \"LOW\"\n"
                "- total_years_exp: 0.0",
            ),
        ]
        slots = "\n\n".join(
            f"Candidate {i + 1} (ID {cid}) — {label}:\n{desc}"
            for i, (cid, (label, desc)) in enumerate(zip(ids, _ip_specs))
        )
        body = f"""
Generate exactly {count} candidates with intentionally incomplete or vague data.

Candidate IDs (use in order): {id_str}
type field: "incomplete_profile"

{slots}
"""

    elif group_type == "duplicate":
        _dup_specs = [
            (
                "Clean resume",
                "- Realistic Indian ML engineer, 5–7 years, working at a top Indian startup\n"
                "- Complete fields, professional tone\n"
                "- github_url: https://github.com/<username>\n"
                "- confidence_level: \"HIGH\"",
            ),
            (
                "LinkedIn export of the same person",
                "- SAME name (but formatted differently — e.g., \"Arjun S. Mehta\" vs \"Arjun Mehta\")\n"
                "- SAME companies, SAME start_date and end_date values (copy exactly)\n"
                "- SAME github_url\n"
                "- ONE fewer skill in the skills array\n"
                "- raw_text worded differently (more informal, slightly shorter)\n"
                "- confidence_level: \"HIGH\"",
            ),
        ]
        slots = "\n\n".join(
            f"Candidate {i + 1} (ID {cid}) — {label}:\n{desc}"
            for i, (cid, (label, desc)) in enumerate(zip(ids, _dup_specs))
        )
        body = f"""
Generate exactly {count} profiles for the SAME person submitted twice.

Candidate IDs (use in order): {id_str}
type field: "duplicate"

{slots}

The pipeline's dedup filter should catch them because of matching companies + GitHub URL.
"""

    elif group_type == "github_active":
        body = f"""
Generate {count} candidates whose resume UNDERSELLS them but who have an active GitHub.

Candidate IDs (use in order): {id_str}
type field: "github_active"

Requirements:
- Short, vague raw_text (write only 2–3 sentences, no detail)
- Skills list: exactly ["Python", "SQL", "Git"] — intentionally generic
- experience summary is vague ("worked on data projects", "contributed to ML pipeline")
- BUT github_url is present and username suggests an active ML developer
  (e.g., https://github.com/<name>-devml or similar active-sounding handle)
- confidence_level: "MEDIUM"
- Locations: Bangalore or remote
- These test whether GitHub enrichment can rescue their low resume score
"""

    elif group_type == "poor_match":
        _pm_specs = [
            (
                "Finance analyst",
                "- Skills: Excel, SQL, Bloomberg Terminal, financial modelling, PowerPoint, VBA\n"
                "- Experience: equity research, investment banking, or corporate finance\n"
                "- Zero Python or ML skills",
            ),
            (
                "Legal professional",
                "- Skills: legal research, contract drafting, compliance, LexisNexis, MS Word\n"
                "- Experience: law firm associate, corporate counsel, compliance officer\n"
                "- No technical skills at all",
            ),
            (
                "Marketing manager",
                "- Skills: SEO, Google Analytics, HubSpot, copywriting, social media management, Canva\n"
                "- Experience: digital marketing, brand management, growth campaigns\n"
                "- No programming",
            ),
            (
                "Civil engineer",
                "- Skills: AutoCAD, STAAD Pro, structural analysis, project management, MS Project\n"
                "- Experience: construction, infrastructure, site management\n"
                "- No data or ML background",
            ),
        ]
        slots = "\n\n".join(
            f"Candidate {i + 1} (ID {cid}): {label}\n{desc}"
            for i, (cid, (label, desc)) in enumerate(zip(ids, _pm_specs))
        )
        body = f"""
Generate exactly {count} candidates from WRONG domains — they should rank LAST for an ML Engineer role.

Candidate IDs (use in order): {id_str}
type field: "poor_match"

{slots}

All {count}: Indian names, Indian companies in their domains, confidence_level "HIGH",
realistic Indian locations.
"""

    else:
        body = f"Generate {count} candidates of type '{group_type}'.\nIDs: {id_str}\n"

    return f"{target_jd}\n{body}\n{_SCHEMA_RULES}"


# ---------------------------------------------------------------------------
# 7. Per-group generation
# ---------------------------------------------------------------------------

def generate_group(
    client: OpenAI,
    group_type: str,
    count: int,
    start_id: int,
) -> list[dict]:
    """
    Call GPT-4o-mini once for one candidate group, parse, validate, and
    apply any group-specific structural fixups.

    Falls back to a blank-slate placeholder if all retries fail.
    """
    ids = [f"C{str(i).zfill(3)}" for i in range(start_id, start_id + count)]
    prompt = build_group_prompt(group_type, count, ids)

    candidates_raw: list[dict] = []
    for attempt in range(3):
        raw_text = call_llm(client, prompt, max_tokens=4096)
        candidates_raw = parse_json_response(raw_text)

        if len(candidates_raw) >= count:
            candidates_raw = candidates_raw[:count]
            break
        if candidates_raw:
            print(f"    ⚠ Got {len(candidates_raw)} / {count} — padding with blanks")
            break
        print(f"    ⚠ Parse failed, attempt {attempt + 1}/3")

    # Pad to expected count if LLM returned too few
    while len(candidates_raw) < count:
        candidates_raw.append({})

    # Validate and normalise each candidate
    result = [
        validate_candidate(raw, cid, group_type)
        for raw, cid in zip(candidates_raw, ids)
    ]

    # Structural fixups for groups that require enforced null / modified fields
    if group_type == "incomplete_profile":
        _fixup_incomplete_profile(result)
    elif group_type == "duplicate":
        _fixup_duplicate(result)
    elif group_type == "github_active":
        _fixup_github_active(result)

    return result


# ---------------------------------------------------------------------------
# 8. JD generation
# ---------------------------------------------------------------------------

_JD_PROMPTS: dict[str, str] = {
    "jd_senior_ml.txt": """
Write a detailed, high-quality job description for a Senior ML Engineer at an Indian
AI product startup based in Bangalore. Quality target: HIGH (85+/100 on a completeness scale).

Include:
- Company background: 2–3 sentences about a fictional AI-powered SaaS startup (give it a name)
- Role overview: what the person will own, the business impact
- Responsibilities: 8–10 concrete technical bullet points
- Must-have requirements: Python, PyTorch or TensorFlow, ML model training, 3+ years exp
- Nice-to-have: MLOps, Kubernetes, distributed training, Triton, Weights & Biases
- Seniority: ideal 5–8 years; senior IC contributor
- Location: Bangalore (hybrid — 3 days in office), open to remote for exceptional candidates
- Culture & values section (4–5 bullet points)
- Compensation: "Competitive, ESOPs included"
- How to apply: 1 line

Write 400–500 words. Professional tone. Use section headers. Make it specific and compelling.
""",
    "jd_data_scientist.txt": """
Write a medium-quality job description for a Data Scientist role at a fintech company
in Bangalore or Chennai. Quality target: MEDIUM (50–70/100 completeness).

Include:
- Short company background (1–2 vague sentences)
- Role overview (broad, not super specific)
- Responsibilities: 5–6 bullet points (some vague, some specific)
- Requirements: Python mentioned, framework preference unclear, experience range 3–6 years
- Some overlap with ML engineering; unclear if this is a modelling or analytics role
- Location: Bangalore or Chennai

150–250 words. Moderate professionalism. Some requirements are clear, some are fuzzy.
""",
    "jd_mlops_vague.txt": """
Write a deliberately vague, low-quality job description for a "Machine Learning Operations"
role. Quality target: LOW (<50/100 completeness — tests the JD quality-gate fallback).

Characteristics:
- Very short (80–120 words)
- Heavy on buzzwords: "AI", "ML", "cloud", "scalable", "data-driven", "innovation"
- No specific technologies named (just "ML tools", "cloud platforms")
- Unclear seniority ("some experience preferred")
- Company description is generic ("a fast-growing tech company")
- No clear responsibilities
- Requirements are vague ("knowledge of ML is a plus")

This JD should fail a quality check and trigger the industry-skills fallback in the pipeline.
""",
}


def generate_jd(client: OpenAI, filename: str) -> str:
    """Call GPT-4o-mini to generate one job description and return the text."""
    prompt = _JD_PROMPTS[filename]
    return call_llm(client, prompt, max_tokens=1200)


# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------

def save_candidates(candidates: list[dict]) -> None:
    """Write all 30 candidates to data/synthetic/candidates.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_FILE.open("w", encoding="utf-8") as fh:
        json.dump(candidates, fh, indent=2, ensure_ascii=False)


def save_metadata(candidates: list[dict]) -> None:
    """Write coverage summary to data/synthetic/metadata.json."""
    coverage: dict[str, int] = {}
    for c in candidates:
        label = str(c.get("type") or "unknown")
        coverage[label] = coverage.get(label, 0) + 1

    metadata = {
        "total_candidates": len(candidates),
        "coverage": coverage,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_jd": "Senior ML Engineer",
        "notes": "Synthetic dataset for HireSynth pipeline testing",
    }
    with METADATA_FILE.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 10. Coverage report
# ---------------------------------------------------------------------------

def print_coverage_report(candidates: list[dict]) -> None:
    """Print a per-group coverage summary to stdout."""
    actual: dict[str, int] = {}
    for c in candidates:
        label = str(c.get("type") or "unknown")
        actual[label] = actual.get(label, 0) + 1

    print()
    for i, (group_type, expected, _) in enumerate(GROUP_SPECS, start=1):
        count = actual.get(group_type, 0)
        mark = "✓" if count == expected else "⚠"
        print(f"  {mark} Group {i:2d} — {group_type:<25}  {count} candidates")

    total = len(candidates)
    mark = "✓" if total == 30 else "⚠"
    print(f"\n  {mark} Total: {total} candidates saved to {OUTPUT_DIR}/")
    print()


# ---------------------------------------------------------------------------
# 11. Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate LLM calls, validation, and file writes for the full dataset."""
    print("\n  HireSynth Dataset Generator")
    print("  " + "─" * 45)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Candidates ─────────────────────────────────────────────────────────
    all_candidates: list[dict] = []
    for group_type, count, start_id in GROUP_SPECS:
        print(f"\n  Generating {count}× {group_type} (C{start_id:03d}–C{start_id + count - 1:03d})...")
        group = generate_group(client, group_type, count, start_id)
        all_candidates.extend(group)
        print(f"  ✓ {group_type}: {len(group)} candidates")

    save_candidates(all_candidates)
    print(f"\n  ✓ Saved {len(all_candidates)} candidates → {CANDIDATES_FILE}")

    # ── Job descriptions ────────────────────────────────────────────────────
    print("\n  Generating job descriptions...")
    jd_quality: dict[str, str] = {
        "jd_senior_ml.txt": "HIGH",
        "jd_data_scientist.txt": "MEDIUM",
        "jd_mlops_vague.txt": "LOW",
    }
    for filename, quality in jd_quality.items():
        print(f"    {filename} ({quality} quality)...")
        jd_text = generate_jd(client, filename)
        (OUTPUT_DIR / filename).write_text(jd_text, encoding="utf-8")
        print(f"  ✓ {filename}")

    # ── Metadata ────────────────────────────────────────────────────────────
    save_metadata(all_candidates)
    print(f"  ✓ Saved metadata → {METADATA_FILE}")

    # ── Coverage report ─────────────────────────────────────────────────────
    print_coverage_report(all_candidates)


if __name__ == "__main__":
    main()
