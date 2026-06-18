# HireSynth — AI Candidate Ranking System
## Project Context for Claude Code

---

## What This Project Is
An intelligent, multi-signal candidate ranking pipeline for a hackathon 
hosted by Redrob AI India. The goal is to rank candidates against a job 
description using semantic understanding, behavioral signals, and LLM 
reasoning — going far beyond keyword matching.

This is NOT a simple search system. It is a 5-layer intelligent ranking 
brain that thinks like a human recruiter.

---

## Tech Stack
- Python 3.11+
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- faiss-cpu for vector search
- rank-bm25 for sparse retrieval
- openai (GPT-4o-mini) for JD parsing + LLM re-ranking
- pandas + numpy for signal computation
- requests for GitHub API enrichment
- python-dotenv for API key management
- asyncio for parallel LLM calls

---

## Project Structure
hiresynth/
├── data/
│   ├── synthetic/
│   │   ├── candidates.json
│   │   ├── jd_senior_ml.txt
│   │   ├── jd_data_scientist.txt
│   │   └── jd_mlops.txt
│   └── ground_truth/
│       └── manual_rankings.json
├── pipeline/
│   ├── layer0_normalizer.py
│   ├── layer1_retrieval.py
│   ├── layer2_scoring.py
│   ├── layer2_github.py
│   ├── layer3_reranker.py
│   └── layer4_evaluation.py
├── utils/
│   ├── schema.py
│   ├── logger.py
│   └── validators.py
├── output/
│   └── ranked_output.json
├── run_pipeline.py
├── generate_dataset.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

---

## The 5-Layer Pipeline

### Layer 0 — Normalization
Two parallel streams:

LEFT: JD Parser
- Takes raw job description text
- Calls GPT-4o-mini to extract structured intent object:
  {must_have, nice_have, seniority, domain, culture_signals,
   implicit_needs, retrieval_mode, weights}
- Runs JD quality gate — scores completeness 0-100
- If quality LOW (<50): use industry skill set fallback
- Detects multi-role JDs — splits into primary + secondary tracks
- Output: validated structured JD intent object with defaults

RIGHT: Resume Normalizer
- Pass 1: Rule-based extraction (regex + spaCy)
- Pass 2: LLM fallback for missing/unstructured fields
- Output: nullable schema per candidate:
  {candidate_id, name, skills, experience[], education,
   location, total_years_exp, trajectory_label, github_url,
   raw_text, confidence_level, language_detected}
- Confidence tagging: HIGH / MEDIUM / LOW
- Dedup filter: embedding cosine sim > 0.92 OR
  (same exp_years + same top3_skills + same institution) → flag duplicate
- Generate deterministic candidate_id if missing:
  md5(name + email + first_employer)[:8]
- Input validation: min length checks, detect scan-only PDFs,
  normalize all date formats, ensure unique IDs

### Layer 1 — Semantic Recall
- Dense search: FAISS with all-MiniLM-L6-v2 embeddings
  → Build index ONCE offline, reuse for all JDs
  → Embed candidate profiles at index build time
- Sparse search: BM25 on skills + job titles
  → Chunked processing if candidates > 10,000
- RRF Fusion: Reciprocal Rank Fusion combines both lists
  → retrieval_mode = "technical" → weight BM25 higher
  → retrieval_mode = "semantic" → weight dense higher
- Output: top 200 candidates

### Layer 2 — Signal Fusion Scoring
Five signals scored independently then fused:

SKILLS MATCH (Tier 1, base weight 40%)
- Semantic skill expansion via LLM — run ONCE at startup, cache it
- Adaptive threshold: niche role = 0.65, broad role = 0.80
- Score normalized 0-1 before weighting

TRAJECTORY (Tier 1, base weight 20%)
- Compute growth arc from job titles + dates
- Detect exceptions before scoring as negative:
  → Startup gap: gap + "founder/co-founder" keywords → neutral
  → Consulting pattern: repeated senior titles at diff companies → neutral
  → Domain expansion: lateral + new skill cluster → slight positive
- Career changer detection: large education-to-now gap + no exp field
  → trigger LLM extraction before labeling as fresher

RECENCY (Tier 2, base weight 15%)
- Decay multiplier: last 1yr=1.0, 1-3yr=0.85, 3-5yr=0.65, 5+yr=0.4
- Applied to skill scores, not just overall experience

BEHAVIORAL (Tier 2, base weight 15%)
- Activity recency, profile completeness, openness-to-move signals
- Company stage history as intent signal

SENIORITY ALIGNMENT (base weight 10%)
- Bidirectional: detects BOTH underqualified AND overqualified
- Overqualified → flag is_overqualified: true, do not rank #1

FUSION RULES:
- All signals normalize to 0-1 before weighting
- Weights adjust dynamically based on JD parser output
- Location: soft boost only (T3), never a penalty
- Low-match gate: if top candidate < 40/100 → low_match warning
- Hidden gem filter: skills >= 60% AND trajectory=high_growth
  AND final_score 60-79 → hidden_gem tier (cap at 5)

SCORE NORMALIZATION CONTRACT:
- raw_score → normalized_score (0-1) → weighted_contribution
- Output breakdown shows all three per signal
- Location can only boost, never reduce score

### GitHub Enrichment (between Layer 2 and Layer 3)
- Trigger: only for top 50 candidates
- Extract GitHub URL from Layer 0 normalizer output
- Call GitHub API (unauthenticated, 60 req/hr limit):
  GET /users/{username} → repos, stars, followers, created_at
  GET /users/{username}/events → last_push date
- Hard timeout: 2 seconds per call
- On timeout/fail → github_enrichment: null, weight → 0
- Score from top 3 languages (not just top 1)
  Map equivalents: Jupyter Notebook → Python, TypeScript → JavaScript
- GitHub score signals:
  recency (last push < 30 days), volume (repos > 20),
  credibility (stars > 50), language_match, account_age
- Weight: 5-8% of final score
- BOOST ONLY — never penalize absence of GitHub
- Flag: github_verified: true/false in output

### Layer 3 — LLM Re-ranking
- GPT-4o-mini, temperature=0 for determinism
- Async parallel calls, semaphore limit = 10 concurrent
- Cap candidate summary at 500 tokens before sending:
  Keep: skills, last 3 roles (1 sentence each), education,
        github_score, trajectory_label
  Discard: older roles, certifications, full role descriptions
- Candidate anonymization: strip names → use Candidate_ID only
- Structured reasoning card contract (enforce via prompt):
  Line 1: Top strength (1 sentence)
  Line 2: Key gap or risk (1 sentence)
  Line 3: Recruiter recommendation (1 sentence)
  Score: X/10
- Validate output format before writing to schema
- Conflict checker (bidirectional):
  if llm_score vs fusion_score delta > 15 points → conflict flag
  if llm_says_strong AND signal < 50 → reasoning_verified: false
  if llm_says_weak AND signal > 80 → reasoning_verified: false
  Resolution: fusion score wins on conflict
- API fallback: on failure → use L2 score, flag llm_rerank: false
- Rate limit handling: asyncio.Semaphore(10)

### Layer 4 — Evaluation + Audit
- NDCG evaluator: 30-sample manual ground truth
  Report with confidence interval, note proxy metric limitation
- Bias audit (preventive, not just post-hoc):
  Field level: name, gender, prestige excluded from all scoring
  Prompt level: explicit review for demographic language
  LLM receives only Candidate_ID, never name
- Pipeline audit log per run:
  {run_id, embedding_model, jd_quality_score,
   skill_threshold_used, signal_weights,
   candidates_at_each_stage, run_hash}
- Run hash: md5(jd_text + sorted(candidate_ids))
  → Same inputs always produce same hash

---

## Locked Output Schema
```json
{
  "metadata": {
    "job_id": "JD_001",
    "job_title": "Senior ML Engineer",
    "jd_quality_score": 82,
    "jd_quality_level": "HIGH",
    "total_candidates_processed": 30,
    "total_candidates_shortlisted": 10,
    "low_match_warning": false,
    "pipeline_version": "1.0.0",
    "generated_at": "2026-06-12T10:30:00Z",
    "run_hash": "a3f8b2c1",
    "exit_status": "EXIT_SUCCESS"
  },
  "shortlist": [
    {
      "rank": 1,
      "candidate_id": "C047",
      "scores": {
        "final_score": 84,
        "signal_fusion_score": 81,
        "llm_rerank_score": 87,
        "score_conflict": false
      },
      "signal_breakdown": {
        "skills_match":    {"raw": 0.80, "normalized": 0.80, "weighted": 32},
        "trajectory":      {"raw": 0.90, "normalized": 0.90, "weighted": 18},
        "recency":         {"raw": 0.70, "normalized": 0.70, "weighted": 10},
        "behavioral":      {"raw": 0.80, "normalized": 0.80, "weighted": 12},
        "seniority_align": {"raw": 1.00, "normalized": 1.00, "weighted": 10},
        "github_boost":    {"raw": 0.74, "normalized": 0.74, "weighted": 4}
      },
      "confidence": {
        "level": "HIGH",
        "fields_present": ["skills","experience","education","location"],
        "fields_inferred": [],
        "fields_missing": []
      },
      "flags": {
        "is_fresher": false,
        "is_overqualified": false,
        "is_duplicate": false,
        "incomplete_profile": false,
        "score_conflict": false,
        "low_match": false,
        "llm_rerank": true,
        "github_verified": true,
        "reasoning_verified": true
      },
      "tier": "strong_match",
      "reasoning": {
        "strength": "...",
        "gap": "...",
        "recommendation": "...",
        "llm_score": 9
      },
      "skills_matched": ["PyTorch", "Python"],
      "skills_missing": ["Kubernetes"],
      "experience_years": 5,
      "trajectory_label": "high_growth",
      "seniority_label": "senior",
      "location_match": "exact"
    }
  ],
  "hidden_gems": [],
  "evaluation": {
    "ndcg_score": 0.91,
    "confidence_interval": "±0.06",
    "eval_set_size": 30,
    "note": "Proxy metric — manual ground truth"
  },
  "audit_log": {},
  "bias_note": "Excluded: name, gender, university prestige, photo"
}
```

---

## Exit States
```python
EXIT_SUCCESS      = "ranked output produced"
EXIT_LOW_MATCH    = "output produced, low confidence warning"
EXIT_NO_CANDIDATES = "pipeline halted: 0 candidates passed retrieval"
EXIT_JD_PARSE_FAIL = "pipeline halted: JD parsing returned empty"
EXIT_API_DOWN     = "output produced using L2 scores only"
```

---

## Synthetic Dataset — 30 Candidate Breakdown
5 strong matches (all fields present)
5 good matches with gaps (missing nice-to-haves)
3 hidden gems (skills 60-75%, high trajectory)
2 freshers (recent graduation, projects section)
2 overqualified (15+ years, mid-level role)
2 career changers (MBA gap or startup gap)
3 incomplete profiles (missing dates/location/skills)
2 duplicates (same person, two resumes)
2 GitHub-active underscorers (weak resume, strong GitHub)
4 poor matches (wrong domain entirely)

---

## Key Design Principles
1. Never penalize missing data — only reward present data
2. Normalize score against available signals, not expected signals
3. GitHub enrichment is boost-only, never a penalty
4. Location is soft boost only, never a penalty
5. Fusion score wins over LLM score on conflict
6. Temperature=0 for all LLM calls for determinism
7. np.random.seed(42) for all random operations
8. All signals normalize to 0-1 before weighting
9. Hidden gem tier capped at 5 candidates
10. Single entry point: python run_pipeline.py

---

## What Makes This Different From Redrob
- Semantic understanding vs keyword matching
- JD intent decomposition vs raw text blob
- Career trajectory scoring vs raw years
- Behavioral signals (nobody else does this)
- Bidirectional seniority (over + underqualified)
- GitHub enrichment for verification
- Per-candidate reasoning cards
- Confidence scoring per candidate
- Hidden gem tier
- Full explainability with signal breakdown

---

## Coding Rules
- Always use python-dotenv for API keys, never hardcode
- Always validate LLM JSON output before using it
- Always set timeout=2 on GitHub API calls
- Always use asyncio.Semaphore(10) for parallel LLM calls
- Always set temperature=0 for LLM re-ranker
- Always set np.random.seed(42)
- Every function needs a docstring
- Every file needs a module-level docstring
- Use dataclasses or TypedDict for all schemas
- Write requirements.txt with pinned versions