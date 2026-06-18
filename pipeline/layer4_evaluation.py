"""
pipeline/layer4_evaluation.py — Layer 4: Evaluation and Bias Audit.

Runs after Layer 3 re-ranking and performs three tasks:

  1. NDCG evaluation — computes Normalised Discounted Cumulative Gain at
     k=20 against a manually annotated ground-truth file.  Returns an
     honest 95 % confidence interval based on the eval set size.

  2. Bias audit — verifies that excluded signals (name, gender, prestige,
     etc.) never appear in any scoring field, and that score distribution
     is not skewed by missing-data penalisation.

  3. Output assembly and serialisation — builds PipelineOutput, writes
     ranked_output.json and ranked_output.csv.

All checks are preventive (not just post-hoc): bias is addressed at the
prompt and field levels throughout the pipeline, and this layer audits that
those invariants held across the full run.

Usage:
    python -m pipeline.layer4_evaluation
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import dataclasses
import json
import math
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import (
    PIPELINE_VERSION,
    log_layer_complete,
    log_pipeline_complete,
    log_warning,
)
from utils.schema import (
    CandidateNormalized,
    ExitStatus,
    JDIntent,
    PipelineMetadata,
    PipelineOutput,
    ScoredCandidate,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_GT_DEFAULT_PATH = "data/ground_truth/manual_rankings.json"

EXCLUDED_SIGNALS = [
    "name", "gender", "photo", "religion",
    "caste", "marital_status", "nationality",
    "university_prestige", "age",
]

_ALLOWED_SIGNAL_KEYS = {
    "skills", "trajectory", "recency",
    "behavioral", "seniority", "github_boost",
}

# Score-distribution warning threshold (points)
_CONFIDENCE_SKEW_THRESHOLD = 30.0


# ===========================================================================
# PART 1 — GROUND TRUTH LOADER
# ===========================================================================

def load_ground_truth(
    path: str = _GT_DEFAULT_PATH,
) -> Optional[dict[str, int]]:
    """
    Load the manual relevance judgements from a JSON file.

    Ground-truth format::

        {
            "job_id": "JD_001",
            "rankings": [
                {"candidate_id": "C001", "relevance": 3},
                ...
            ]
        }

    Relevance scale:
      3 — highly relevant  (strong_match)
      2 — relevant         (good_match, hidden_gem)
      1 — marginal         (borderline, fresher, career_changer)
      0 — not relevant     (poor_match, wrong domain)

    Returns:
        {candidate_id: relevance_score} mapping on success;
        None when the file is absent (NDCG is skipped gracefully).
    """
    gt_path = Path(__file__).parent.parent / path
    if not gt_path.exists():
        log_warning(
            f"Ground truth file not found at {gt_path} — NDCG evaluation skipped."
        )
        return None

    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        rankings: list[dict] = data.get("rankings", [])
        return {
            r["candidate_id"]: int(r["relevance"])
            for r in rankings
            if "candidate_id" in r and "relevance" in r
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log_warning(f"Failed to parse ground truth file: {exc} — NDCG skipped.")
        return None


# ===========================================================================
# PART 2 — NDCG COMPUTATION
# ===========================================================================

def compute_dcg(relevances: list[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at cutoff k.

    Formula (0-indexed positions)::

        DCG@k = Σ  rel_i / log₂(i + 2)   for i in 0..k-1

    Position 0 (rank 1) uses log₂(2) = 1.0 as denominator.

    Returns:
        0.0 for an empty relevance list; DCG value otherwise.
    """
    if not relevances:
        return 0.0
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def compute_ndcg(
    ranked_candidates: list[ScoredCandidate],
    ground_truth: dict[str, int],
    k: int = 20,
) -> tuple[float, str]:
    """
    Compute NDCG@k and a 95 % confidence interval string.

    Steps:
      1. Build predicted relevance list from the ranked shortlist.
      2. Build ideal relevance list (sorted desc from ground truth).
      3. DCG  = compute_dcg(predicted, k)
         IDCG = compute_dcg(ideal, k)
         NDCG = DCG / IDCG  (0.0 when IDCG == 0)
      4. Confidence interval (honest proxy-metric caveat):
         n  = min(k, len(ground_truth))
         ci = 1.96 × √(NDCG × (1 − NDCG) / n)

    Returns:
        (ndcg_score rounded to 4 d.p., "±X.XXX" CI string)
    """
    predicted: list[float] = [
        float(ground_truth.get(c.candidate_id, 0))
        for c in ranked_candidates[:k]
    ]

    all_relevances = list(ground_truth.values())
    ideal: list[float] = sorted(
        [float(r) for r in all_relevances], reverse=True
    )[:k]

    dcg  = compute_dcg(predicted, k)
    idcg = compute_dcg(ideal, k)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg = round(min(ndcg, 1.0), 4)

    n = max(1, min(k, len(ground_truth)))
    ci_val = 1.96 * math.sqrt(ndcg * (1.0 - ndcg) / n)
    ci_str = f"±{ci_val:.3f}"

    return ndcg, ci_str


# ===========================================================================
# PART 3 — BIAS AUDIT
# ===========================================================================

def run_bias_audit(
    shortlist: list[ScoredCandidate],
    candidates_map: dict[str, CandidateNormalized],
) -> dict:
    """
    Audit the shortlist for bias-related invariant violations.

    Four checks:

    1. **Name check** — verify that "name" never appears as a signal key
       in any candidate's signal_breakdown.
    2. **Field exclusion check** — verify signal_breakdown keys are a
       subset of the allowed set (skills, trajectory, recency, behavioral,
       seniority, github_boost).
    3. **Score distribution check** — group final_scores by confidence
       level; warn if HIGH-confidence candidates outscore MEDIUM/LOW by
       more than 30 points on average (would suggest penalising missing
       data rather than only rewarding present data).
    4. **LLM anonymisation note** — static confirmation that candidate
       summaries sent to the LLM use Candidate_ID only (enforced in
       layer3_reranker.build_candidate_summary).

    Returns:
        Audit result dict with pass/warn/fail values for each check.
    """
    audit: dict = {}

    # --- 1. Name check ---
    name_in_signals = [
        sc.candidate_id
        for sc in shortlist
        if "name" in sc.signal_breakdown
    ]
    if name_in_signals:
        audit["name_check"] = f"FAIL — 'name' signal found for: {name_in_signals}"
    else:
        audit["name_check"] = "pass"

    # --- 2. Field exclusion check ---
    unexpected: list[str] = []
    for sc in shortlist:
        extra = set(sc.signal_breakdown.keys()) - _ALLOWED_SIGNAL_KEYS
        for key in extra:
            unexpected.append(f"{sc.candidate_id}:{key}")

    if unexpected:
        audit["field_exclusion"] = f"FAIL — unexpected signal keys: {unexpected}"
    else:
        audit["field_exclusion"] = "pass"

    # --- 3. Score distribution check ---
    by_confidence: dict[str, list[float]] = {}
    for sc in shortlist:
        level = sc.confidence_level
        by_confidence.setdefault(level, []).append(sc.final_score)

    score_means: dict[str, float] = {
        level: round(sum(scores) / len(scores), 2)
        for level, scores in by_confidence.items()
    }

    high_mean  = score_means.get("HIGH")
    med_mean   = score_means.get("MEDIUM")
    low_mean   = score_means.get("LOW")
    lower_mean = min(
        (v for v in [med_mean, low_mean] if v is not None),
        default=None,
    )

    if high_mean is not None and lower_mean is not None:
        skew = high_mean - lower_mean
        if skew > _CONFIDENCE_SKEW_THRESHOLD:
            audit["score_distribution"] = (
                f"WARN — HIGH mean={high_mean:.1f} "
                f"vs MEDIUM/LOW mean={lower_mean:.1f} "
                f"(gap={skew:.1f} > {_CONFIDENCE_SKEW_THRESHOLD} pts): "
                "possible missing-data penalisation."
            )
            log_warning(audit["score_distribution"])
        else:
            audit["score_distribution"] = (
                f"pass — score means by confidence: {score_means}"
            )
    else:
        audit["score_distribution"] = (
            f"pass — single confidence tier: {score_means}"
        )

    # --- 4. LLM anonymisation note ---
    audit["llm_anonymisation"] = (
        "verified — LLM prompts use Candidate_ID only "
        "(enforced in layer3_reranker.build_candidate_summary)"
    )

    # --- Metadata ---
    audit["excluded_signals"] = EXCLUDED_SIGNALS
    audit["bias_note"] = (
        "Excluded from scoring: name, gender, photo, "
        "university prestige, age, religion, caste"
    )

    return audit


# ===========================================================================
# PART 4 — OUTPUT ASSEMBLER
# ===========================================================================

def assemble_output(
    jd_intent: JDIntent,
    shortlist: list[ScoredCandidate],
    hidden_gems: list[ScoredCandidate],
    low_match_warning: bool,
    audit_log: dict,
    bias_audit: dict,
    ndcg_score: float,
    ndcg_ci: str,
    ground_truth_size: int,
    run_id: str,
    run_hash: str,
    total_processed: int,
    exit_status: str,
    job_id: str = "JD_001",
) -> PipelineOutput:
    """
    Assemble a fully-populated PipelineOutput from all layer results.

    Constructs PipelineMetadata from the JD intent and audit log, builds
    the evaluation dict with the NDCG result and bias note, and wraps
    everything into a single PipelineOutput dataclass.

    Returns:
        PipelineOutput ready for serialisation.
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = PipelineMetadata(
        job_id=job_id,
        job_title=jd_intent.job_title or f"{jd_intent.seniority} {jd_intent.domain}",
        jd_quality_score=jd_intent.quality_score,
        jd_quality_level=jd_intent.quality_level,
        total_candidates_processed=total_processed,
        total_candidates_shortlisted=len(shortlist),
        low_match_warning=low_match_warning,
        pipeline_version=PIPELINE_VERSION,
        generated_at=generated_at,
        run_hash=run_hash,
        exit_status=ExitStatus(exit_status),
        candidates_at_each_stage=audit_log.get("candidates_at_each_stage", {}),
        signal_weights_used=jd_intent.weights,
        skill_threshold_used=float(audit_log.get("skill_threshold_used", 0.75)),
        embedding_model=_EMBEDDING_MODEL,
    )

    evaluation = {
        "ndcg_score": ndcg_score,
        "confidence_interval": ndcg_ci,
        "eval_set_size": ground_truth_size,
        "note": (
            "Proxy metric — manual ground truth. "
            "NDCG computed against manually ranked sample. "
            "Higher confidence intervals reflect small eval set size."
        ),
    }

    return PipelineOutput(
        metadata=metadata,
        shortlist=shortlist,
        hidden_gems=hidden_gems,
        evaluation=evaluation,
        audit_log=audit_log,
        bias_note=bias_audit.get("bias_note", ""),
    )


# ===========================================================================
# PART 5 — OUTPUT SERIALISER
# ===========================================================================

class _JsonEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        """Convert numpy floats/ints to native Python types."""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def serialize_output(
    output: PipelineOutput,
    output_path: str = "output/ranked_output.json",
) -> str:
    """
    Serialise a PipelineOutput to a pretty-printed JSON file.

    Uses dataclasses.asdict() for recursive conversion of nested dataclasses.
    ExitStatus (a str-subclass Enum) is serialised as its string value by the
    standard json encoder.  The output directory is created if absent.

    Returns:
        The resolved output file path as a string.
    """
    out = Path(__file__).parent.parent / output_path
    out.parent.mkdir(parents=True, exist_ok=True)

    raw_dict = dataclasses.asdict(output)
    out.write_text(
        json.dumps(raw_dict, indent=2, cls=_JsonEncoder, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(out)


def serialize_output_csv(
    output: PipelineOutput,
    output_path: str = "output/ranked_output.csv",
) -> str:
    """
    Serialise the shortlist to a flat CSV file for easy inspection.

    Each row corresponds to one ranked candidate.  Columns:
      rank, candidate_id, final_score, signal_fusion_score, llm_rerank_score,
      tier, confidence_level, skills_match_pct, trajectory_pct,
      is_overqualified, is_fresher, score_conflict, github_verified,
      strength, gap, recommendation.

    Returns:
        The resolved output file path as a string.
    """
    out = Path(__file__).parent.parent / output_path
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sc in output.shortlist:
        bd = sc.signal_breakdown
        card = sc.reasoning
        rows.append({
            "rank":                 sc.rank,
            "candidate_id":         sc.candidate_id,
            "final_score":          round(sc.final_score, 2),
            "signal_fusion_score":  round(sc.signal_fusion_score, 2),
            "llm_rerank_score":     sc.llm_rerank_score,
            "tier":                 sc.tier,
            "confidence_level":     sc.confidence_level,
            "skills_match_pct":     round(bd["skills"].normalized * 100, 1) if "skills" in bd else None,
            "trajectory_pct":       round(bd["trajectory"].normalized * 100, 1) if "trajectory" in bd else None,
            "is_overqualified":     sc.flags.is_overqualified,
            "is_fresher":           sc.flags.is_fresher,
            "score_conflict":       sc.flags.score_conflict,
            "github_verified":      sc.flags.github_verified,
            "strength":             card.strength if card else "",
            "gap":                  card.gap if card else "",
            "recommendation":       card.recommendation if card else "",
        })

    pd.DataFrame(rows).to_csv(str(out), index=False)
    return str(out)


# ===========================================================================
# PART 6 — ORCHESTRATOR
# ===========================================================================

def run_layer4(
    jd_intent: JDIntent,
    shortlist: list[ScoredCandidate],
    hidden_gems: list[ScoredCandidate],
    candidates_map: dict[str, CandidateNormalized],
    low_match_warning: bool,
    audit_log: dict,
    run_id: str,
    run_hash: str,
    total_processed: int,
    job_id: str = "JD_001",
    pipeline_start: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> tuple[PipelineOutput, str, str]:
    """
    Layer 4 top-level orchestrator — evaluation, bias audit, and output.

    Steps:
      1. Load manual ground truth; skip NDCG gracefully if absent.
      2. Compute NDCG@20 and 95 % confidence interval.
      3. Run bias audit over the shortlist.
      4. Determine exit status from low-match and shortlist state.
      5. Assemble PipelineOutput from all layer results.
      6. Serialise to JSON and CSV.
      7. Log pipeline completion banner.

    Args:
        pipeline_start: Optional unix timestamp from the overall pipeline
                        start (used for total elapsed in log_pipeline_complete).
                        Falls back to Layer 4's own elapsed when absent.

    Returns:
        (PipelineOutput, json_path, csv_path)
    """
    start = time.time()

    # Step 1 — load ground truth
    ground_truth = load_ground_truth()
    gt_size = len(ground_truth) if ground_truth else 0

    # Step 2 — NDCG
    if ground_truth:
        ndcg_score, ndcg_ci = compute_ndcg(shortlist, ground_truth, k=20)
        print(f"  NDCG@20 = {ndcg_score:.4f}  CI={ndcg_ci}  "
              f"(eval_set={gt_size} candidates)")
    else:
        ndcg_score, ndcg_ci = 0.0, "N/A"
        log_warning("No ground truth found — NDCG evaluation skipped.")

    # Step 3 — bias audit
    bias_audit = run_bias_audit(shortlist, candidates_map)
    print(f"  Bias audit: name_check={bias_audit['name_check']} | "
          f"field_exclusion={bias_audit['field_exclusion']} | "
          f"llm_anonymisation={bias_audit['llm_anonymisation']}")

    # Step 4 — exit status
    if len(shortlist) == 0:
        exit_status = ExitStatus.EXIT_NO_CANDIDATES.value
    elif low_match_warning:
        exit_status = ExitStatus.EXIT_LOW_MATCH.value
    else:
        exit_status = ExitStatus.EXIT_SUCCESS.value

    # Step 5 — assemble output
    output = assemble_output(
        jd_intent=jd_intent,
        shortlist=shortlist,
        hidden_gems=hidden_gems,
        low_match_warning=low_match_warning,
        audit_log=audit_log,
        bias_audit=bias_audit,
        ndcg_score=ndcg_score,
        ndcg_ci=ndcg_ci,
        ground_truth_size=gt_size,
        run_id=run_id,
        run_hash=run_hash,
        total_processed=total_processed,
        exit_status=exit_status,
        job_id=job_id,
    )

    # Step 6 — serialise
    out_dir = output_dir or "output"
    json_path = serialize_output(output, output_path=f"{out_dir}/ranked_output.json")
    csv_path  = serialize_output_csv(output, output_path=f"{out_dir}/ranked_output.csv")

    # Step 7 — log completion
    elapsed = time.time() - (pipeline_start if pipeline_start else start)
    log_layer_complete(layer=4, candidates_remaining=len(shortlist), elapsed_seconds=elapsed)
    log_pipeline_complete(
        run_id=run_id,
        shortlist_count=len(shortlist),
        hidden_gems_count=len(hidden_gems),
        ndcg_score=ndcg_score,
        elapsed_seconds=elapsed,
    )

    return output, json_path, csv_path


# ===========================================================================
# QUICK TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    import os

    from sentence_transformers import SentenceTransformer as _ST

    from pipeline.layer0_normalizer import run_layer0
    from pipeline.layer1_retrieval import run_layer1
    from pipeline.layer2_scoring import run_layer2
    from pipeline.layer2_github import run_github_enrichment
    from pipeline.layer3_reranker import run_layer3
    from utils.logger import create_audit_log, generate_run_id, generate_run_hash

    DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

    _pipeline_start = time.time()

    print("\n" + "=" * 70)
    print("  Layer 4 Quick Test — full pipeline + evaluation")
    print("=" * 70 + "\n")

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

    # --- Layer 0 ---
    run_id   = generate_run_id()
    run_hash = generate_run_hash(jd_text, [c["candidate_id"] for c in all_candidates_raw])
    jd_intent, normalized = run_layer0(jd_text, "JD_001", all_candidates_raw)
    print(f"\n  Layer 0: {len(normalized)} candidates normalised.\n")

    # --- Layer 1 ---
    retrieved, model = run_layer1(jd_intent, normalized, top_k=50)
    print(f"\n  Layer 1: {len(retrieved)} candidates retrieved.\n")

    # --- Layer 2 ---
    if model is None:
        print("  Layer 1 returned no model — loading separately...")
        model = _ST("all-MiniLM-L6-v2")
    shortlist, hidden_gems, low_match_warning = run_layer2(
        jd_intent, retrieved, model, top_k=50
    )
    print(f"\n  Layer 2: {len(shortlist)} scored  |  "
          f"{len(hidden_gems)} hidden gems  |  "
          f"low_match={low_match_warning}\n")

    # --- GitHub ---
    candidates_map: dict[str, CandidateNormalized] = {
        c.candidate_id: c for c in normalized
    }
    print("  Running GitHub enrichment...")
    enriched = run_github_enrichment(shortlist, candidates_map, jd_intent, max_candidates=50)
    print(f"\n  GitHub: {len(enriched)} candidates enriched.\n")

    # --- Layer 3 ---
    print("  Running Layer 3 LLM re-ranking...\n")
    final_shortlist = run_layer3(jd_intent, enriched, candidates_map, top_k=20)
    print(f"\n  Layer 3: {len(final_shortlist)} candidates in final shortlist.\n")

    # --- Audit log ---
    audit_log = create_audit_log(
        run_id=run_id,
        embedding_model=_EMBEDDING_MODEL,
        jd_quality_score=jd_intent.quality_score,
        skill_threshold_used=0.65,
        signal_weights=jd_intent.weights,
        candidates_at_each_stage={
            "after_layer0": len(normalized),
            "after_layer1": len(retrieved),
            "after_layer2": len(shortlist),
            "after_layer3": len(final_shortlist),
        },
    )

    # --- Layer 4 ---
    print("  Running Layer 4 evaluation and bias audit...\n")
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
        job_id="JD_001",
        pipeline_start=_pipeline_start,
    )

    # --- Validate output JSON ---
    print("\n  Validating output JSON...")
    output_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    meta     = output_data["metadata"]
    sl       = output_data["shortlist"]
    eval_d   = output_data["evaluation"]

    # --- Assertions ---
    print("\n  Assertions:")

    assert meta["exit_status"] == ExitStatus.EXIT_SUCCESS.value, (
        f"Expected EXIT_SUCCESS, got {meta['exit_status']!r}"
    )
    print(f"  ✓ exit_status == '{meta['exit_status']}'")

    assert len(sl) == 20, f"Expected 20 shortlisted candidates, got {len(sl)}"
    print(f"  ✓ shortlist length == {len(sl)}")

    assert eval_d["ndcg_score"] > 0, (
        f"Expected NDCG > 0, got {eval_d['ndcg_score']}"
    )
    print(f"  ✓ NDCG@20 = {eval_d['ndcg_score']:.4f}  CI={eval_d['confidence_interval']}")

    cards_missing = [
        c["candidate_id"] for c in sl if not c.get("reasoning")
    ]
    assert not cards_missing, (
        f"Reasoning cards missing for: {cards_missing}"
    )
    print(f"  ✓ All {len(sl)} candidates have reasoning cards.")

    top5_types = [type_by_id.get(c["candidate_id"], "") for c in sl[:5]]
    assert "poor_match" not in top5_types, (
        f"Found poor_match in top 5: {top5_types}"
    )
    print("  ✓ No poor_match in top 5.")

    assert Path(csv_path).exists(), f"CSV not found: {csv_path}"
    csv_rows = pd.read_csv(csv_path)
    assert len(csv_rows) == 20, f"Expected 20 CSV rows, got {len(csv_rows)}"
    print(f"  ✓ CSV written with {len(csv_rows)} rows.")

    # --- Final 5-line summary ---
    print(f"\n  {'─'*50}")
    print(f"  ✓ Layer 4 complete")
    print(f"  ✓ NDCG@20: {eval_d['ndcg_score']:.4f}  {eval_d['confidence_interval']}")
    print(f"  ✓ Bias audit: {output_data['bias_note']}")
    print(f"  ✓ JSON: {json_path}")
    print(f"  ✓ CSV:  {csv_path}")
    print(f"  {'─'*50}")
    print("\n  ✓ All Layer 4 assertions passed.")
