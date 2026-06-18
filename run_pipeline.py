"""
HireSynth — single entry point.

Usage:
    python run_pipeline.py
    python run_pipeline.py --jd data/synthetic/jd_data_scientist.txt
    python run_pipeline.py --jd data/synthetic/jd_senior_ml.txt \
                           --candidates data/synthetic/candidates.json \
                           --job-id JD_001 --output-dir output --top-k 20

The pipeline runs 5 layers end-to-end:
    L0  Normalise JD + candidates (GPT-4o-mini + rule-based)
    L1  Hybrid retrieval (FAISS dense + BM25 sparse + RRF fusion)
    L2  Multi-signal fusion scoring (5 signals)
    L2.5 GitHub enrichment (boost-only, top 50)
    L3  LLM re-ranking (GPT-4o-mini, async, Semaphore=10)
    L4  NDCG evaluation + bias audit + JSON/CSV output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

np.random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent


def _load_jd_text(jd_path: Path) -> str:
    """Read a JD file and return its full text, flattening JSON if needed."""
    raw = jd_path.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(raw)
        jd_obj = obj.get("job_description", obj)
        parts: list[str] = []
        for key in (
            "role_overview", "responsibilities", "must_have_requirements",
            "nice_to_have", "preferred_qualifications", "technical_requirements",
            "compensation_and_benefits", "about_the_role",
        ):
            val = jd_obj.get(key)
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, list):
                parts.extend(str(v) for v in val)
        return "\n".join(parts) or json.dumps(jd_obj)
    except (json.JSONDecodeError, AttributeError):
        return raw


def _ensure_output_dir(output_dir: str) -> None:
    """Create the output directory if it does not exist."""
    Path(_PROJECT_ROOT / output_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="HireSynth — AI candidate ranking pipeline",
    )
    parser.add_argument(
        "--jd",
        default="data/synthetic/jd_senior_ml.txt",
        help="Path to job description file (.txt or .json). Default: jd_senior_ml.txt",
    )
    parser.add_argument(
        "--candidates",
        default="data/synthetic/candidates.json",
        help="Path to candidates JSON file. Default: data/synthetic/candidates.json",
    )
    parser.add_argument(
        "--job-id",
        default="JD_001",
        dest="job_id",
        help="Job ID tag written to output metadata. Default: JD_001",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        dest="output_dir",
        help="Directory for ranked_output.json and ranked_output.csv. Default: output",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="Final shortlist size (post Layer 3). Default: 20",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        default=False,
        help="Skip GitHub enrichment (faster runs, no API calls to github.com)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate all five pipeline layers end-to-end.

    Returns:
        0 on success (EXIT_SUCCESS or EXIT_LOW_MATCH).
        1 on hard failure (EXIT_NO_CANDIDATES, EXIT_JD_PARSE_FAIL, EXIT_API_DOWN).
    """
    # Step 1 — args + env
    args = _parse_args()
    load_dotenv()

    jd_path        = _PROJECT_ROOT / args.jd
    candidates_path = _PROJECT_ROOT / args.candidates
    job_id         = args.job_id
    output_dir     = args.output_dir
    top_k          = args.top_k
    skip_github    = args.no_github

    if not jd_path.exists():
        print(f"[ERROR] JD file not found: {jd_path}", file=sys.stderr)
        return 1
    if not candidates_path.exists():
        print(f"[ERROR] Candidates file not found: {candidates_path}", file=sys.stderr)
        return 1

    _ensure_output_dir(output_dir)
    pipeline_start = time.time()

    # Step 2 — lazy imports (after dotenv so OPENAI_API_KEY is available)
    from sentence_transformers import SentenceTransformer as _ST

    from pipeline.layer0_normalizer import run_layer0
    from pipeline.layer1_retrieval import run_layer1
    from pipeline.layer2_github import run_github_enrichment
    from pipeline.layer2_scoring import run_layer2
    from pipeline.layer3_reranker import run_layer3
    from pipeline.layer4_evaluation import run_layer4
    from utils.logger import create_audit_log, generate_run_hash, generate_run_id
    from utils.schema import ExitStatus

    _EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Step 3 — load JD text
    jd_text = _load_jd_text(jd_path)
    if not jd_text.strip():
        print("[ERROR] JD file is empty after parsing.", file=sys.stderr)
        return 1

    # Step 4 — load candidates
    all_candidates_raw: list[dict] = json.loads(
        candidates_path.read_text(encoding="utf-8")
    )
    if not all_candidates_raw:
        print("[ERROR] Candidates file is empty.", file=sys.stderr)
        return 1

    # Step 5 — run IDs
    run_id   = generate_run_id()
    run_hash = generate_run_hash(jd_text, [c["candidate_id"] for c in all_candidates_raw])

    print("\n" + "=" * 65)
    print(f"  HireSynth Pipeline  |  run_id={run_id}  |  job={job_id}")
    print(f"  JD: {jd_path.name}   |  candidates: {len(all_candidates_raw)}")
    print("=" * 65 + "\n")

    # Step 6 — Layer 0: normalise JD + candidates
    jd_intent, normalized_candidates = run_layer0(jd_text, job_id, all_candidates_raw)

    if jd_intent is None:
        print("[ERROR] JD parsing returned empty — aborting.", file=sys.stderr)
        return 1

    print(f"\n  L0 complete: {len(normalized_candidates)} candidates normalised.\n")

    if not normalized_candidates:
        print("[ERROR] Zero candidates survived Layer 0 — aborting.", file=sys.stderr)
        return 1

    # Step 7 — Layer 1: hybrid retrieval; returns (candidates, model)
    retrieved, model = run_layer1(jd_intent, normalized_candidates, top_k=200)

    if model is None:
        print("  [warn] Layer 1 returned no model — loading separately...")
        model = _ST(_EMBEDDING_MODEL)

    print(f"\n  L1 complete: {len(retrieved)} candidates retrieved.\n")

    if not retrieved:
        print("[ERROR] Layer 1 retrieved 0 candidates — aborting.", file=sys.stderr)
        return 1

    # Step 8 — Layer 2: multi-signal fusion scoring (reuse model from L1)
    shortlist, hidden_gems, low_match_warning = run_layer2(
        jd_intent, retrieved, model, top_k=50
    )

    print(
        f"\n  L2 complete: {len(shortlist)} scored  |  "
        f"{len(hidden_gems)} hidden gems  |  "
        f"low_match={low_match_warning}\n"
    )

    # Step 9 — Layer 2.5: GitHub enrichment (top 50, boost-only)
    candidates_map = {c.candidate_id: c for c in normalized_candidates}
    if skip_github:
        print(f"\n  L2.5 skipped: --no-github flag set.\n")
    else:
        shortlist = run_github_enrichment(
            shortlist, candidates_map, jd_intent, max_candidates=50
        )
        print(f"\n  L2.5 complete: GitHub enrichment applied.\n")

    # Step 10 — Layer 3: LLM re-ranking (async, Semaphore=10, temperature=0)
    final_shortlist = run_layer3(jd_intent, shortlist, candidates_map, top_k=top_k)
    print(f"\n  L3 complete: {len(final_shortlist)} candidates in final shortlist.\n")

    # Step 11 — Layer 4: evaluation + bias audit + serialise output
    audit_log = create_audit_log(
        run_id=run_id,
        embedding_model=_EMBEDDING_MODEL,
        jd_quality_score=jd_intent.quality_score,
        skill_threshold_used=0.65,
        signal_weights=jd_intent.weights,
        candidates_at_each_stage={
            "after_layer0": len(normalized_candidates),
            "after_layer1": len(retrieved),
            "after_layer2": len(shortlist),
            "after_layer3": len(final_shortlist),
        },
    )

    output, json_path, csv_path = run_layer4(
        jd_intent=jd_intent,
        shortlist=final_shortlist,
        hidden_gems=hidden_gems,
        candidates_map=candidates_map,
        low_match_warning=low_match_warning,
        audit_log=audit_log,
        run_id=run_id,
        run_hash=run_hash,
        total_processed=len(all_candidates_raw),
        job_id=job_id,
        pipeline_start=pipeline_start,
        output_dir=output_dir,
    )

    print(f"\n  Output written to:")
    print(f"    JSON → {json_path}")
    print(f"    CSV  → {csv_path}\n")

    exit_status = output.metadata.exit_status
    hard_failures = {
        ExitStatus.EXIT_NO_CANDIDATES.value,
        ExitStatus.EXIT_JD_PARSE_FAIL.value,
    }
    return 1 if exit_status in hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
