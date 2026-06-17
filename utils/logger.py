"""
Pipeline audit logger for HireSynth.

Records run metadata, stage progress, and completion summaries to stdout
for reproducibility and transparency. Every pipeline run produces a
structured audit log dict plus human-readable console output.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict

from utils.schema import PipelineMetadata  # noqa: F401 — imported for type context


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "1.0.0"

_BANNER_WIDTH = 47
_DIVIDER = "═" * _BANNER_WIDTH


# ---------------------------------------------------------------------------
# 1. Run ID generation
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """
    Generate a unique, human-readable run identifier.

    Format: 'run_YYYYMMDD_HHMMSS' using UTC time.
    Repeated calls within the same second produce the same ID, so callers
    should generate once per pipeline invocation and pass it through.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


# ---------------------------------------------------------------------------
# 2. Reproducibility hash
# ---------------------------------------------------------------------------

def generate_run_hash(jd_text: str, candidate_ids: list) -> str:
    """
    Produce a deterministic 8-character hex fingerprint for a pipeline run.

    Hash input: UTF-8 bytes of jd_text concatenated with the sorted,
    space-joined candidate_ids string.  Identical inputs always yield the
    same hash, enabling diff detection across re-runs.
    """
    sorted_ids = " ".join(sorted(candidate_ids))
    payload = (jd_text + sorted_ids).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:8]


# ---------------------------------------------------------------------------
# 3. Audit log creation
# ---------------------------------------------------------------------------

def create_audit_log(
    run_id: str,
    embedding_model: str,
    jd_quality_score: float,
    skill_threshold_used: float,
    signal_weights: Dict[str, float],
    candidates_at_each_stage: Dict[str, int],
) -> dict:
    """
    Build the structured audit-log dict written to every output file.

    The 'candidates_at_each_stage' dict is normalised to the four expected
    keys (after_layer0 … after_layer3), defaulting missing keys to 0.
    The 'generated_at' timestamp is captured at call time in UTC ISO-8601.
    """
    stage_keys = ["after_layer0", "after_layer1", "after_layer2", "after_layer3"]
    normalised_stages = {k: int(candidates_at_each_stage.get(k, 0)) for k in stage_keys}

    return {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model": embedding_model,
        "jd_quality_score": jd_quality_score,
        "skill_threshold_used": skill_threshold_used,
        "signal_weights": signal_weights,
        "candidates_at_each_stage": normalised_stages,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# 4. Start banner
# ---------------------------------------------------------------------------

def log_pipeline_start(run_id: str, jd_title: str, total_candidates: int) -> None:
    """
    Print a human-readable start banner to stdout.

    Emitted once at the top of every pipeline run so operators can identify
    the run at a glance in logs or CI output.
    """
    print(_DIVIDER)
    print(f"  HireSynth Pipeline v{PIPELINE_VERSION}")
    print(f"  Run ID:     {run_id}")
    print(f"  JD:         {jd_title}")
    print(f"  Candidates: {total_candidates}")
    print(_DIVIDER)


# ---------------------------------------------------------------------------
# 5. Per-layer completion line
# ---------------------------------------------------------------------------

def log_layer_complete(
    layer: int,
    candidates_remaining: int,
    elapsed_seconds: float,
) -> None:
    """
    Print a single progress line after each pipeline layer finishes.

    Format: ✓ Layer N complete — M candidates | X.Xs
    """
    print(
        f"  ✓ Layer {layer} complete "
        f"— {candidates_remaining} candidates | {elapsed_seconds:.1f}s"
    )


# ---------------------------------------------------------------------------
# 6. Completion summary
# ---------------------------------------------------------------------------

def log_pipeline_complete(
    run_id: str,
    shortlist_count: int,
    hidden_gems_count: int,
    ndcg_score: float,
    elapsed_seconds: float,
) -> None:
    """
    Print the final completion summary banner to stdout.

    Includes shortlist size, hidden-gem count, NDCG score, and wall-clock
    time for the full run.
    """
    print(_DIVIDER)
    print(f"  ✓ HireSynth complete")
    print(f"  Run ID:      {run_id}")
    print(f"  Shortlisted: {shortlist_count} candidates")
    print(f"  Hidden gems: {hidden_gems_count} candidates")
    print(f"  NDCG:        {ndcg_score:.2f}")
    print(f"  Total time:  {elapsed_seconds:.1f}s")
    print(_DIVIDER)


# ---------------------------------------------------------------------------
# 7. Warning line
# ---------------------------------------------------------------------------

def log_warning(message: str) -> None:
    """Print a ⚠ WARNING line to stdout."""
    print(f"  ⚠ WARNING: {message}")


# ---------------------------------------------------------------------------
# 8. Error line
# ---------------------------------------------------------------------------

def log_error(message: str) -> None:
    """Print a ✗ ERROR line to stdout."""
    print(f"  ✗ ERROR: {message}")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # Simulate what a full pipeline run would emit to the console.

    run_id = generate_run_id()
    jd_text = (
        "We are hiring a Senior ML Engineer with expertise in Python, PyTorch, "
        "and large-scale distributed training on GPU clusters."
    )
    candidate_ids = [f"C{str(i).zfill(3)}" for i in range(1, 31)]

    run_hash = generate_run_hash(jd_text, candidate_ids)

    log_pipeline_start(run_id, "Senior ML Engineer", total_candidates=30)

    # Simulate layer timings
    for layer, (remaining, elapsed) in enumerate(
        [(28, 1.2), (200, 2.4), (50, 5.8), (10, 4.1)]
    ):
        time.sleep(0.05)  # tiny pause so demo looks sequential
        log_layer_complete(layer, remaining, elapsed)

    log_warning("3 candidates had missing experience dates — using LLM fallback.")
    log_warning("GitHub API rate limit approaching (55 / 60 requests used).")
    log_error("Candidate C019 resume is a scan-only PDF — skipped.")

    audit = create_audit_log(
        run_id=run_id,
        embedding_model="all-MiniLM-L6-v2",
        jd_quality_score=85.0,
        skill_threshold_used=0.72,
        signal_weights={
            "skills": 0.40,
            "trajectory": 0.20,
            "recency": 0.15,
            "behavioral": 0.15,
            "seniority": 0.10,
        },
        candidates_at_each_stage={
            "after_layer0": 28,
            "after_layer1": 200,
            "after_layer2": 50,
            "after_layer3": 10,
        },
    )

    log_pipeline_complete(
        run_id=run_id,
        shortlist_count=10,
        hidden_gems_count=3,
        ndcg_score=0.91,
        elapsed_seconds=13.5,
    )

    print()
    print("  Audit log dict:")
    import json
    print(json.dumps(audit, indent=4))
    print()
    print(f"  Run hash: {run_hash}")
    print(f"  Same hash on re-run: {generate_run_hash(jd_text, candidate_ids) == run_hash}")
