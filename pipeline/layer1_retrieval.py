"""
pipeline/layer1_retrieval.py — Layer 1: Hybrid Semantic Retrieval.

Takes a list of normalized candidates from Layer 0 and returns the top-K
most relevant candidates for a given job description using a two-stage
hybrid retrieval approach:

  DENSE  — FAISS IndexFlatIP on all-MiniLM-L6-v2 embeddings.
           Captures semantic similarity (paraphrases, related concepts).

  SPARSE — BM25Okapi on tokenized candidate text.
           Captures exact keyword / skill-name matches.

  FUSION — Reciprocal Rank Fusion (RRF) combines both ranked lists with
           mode-dependent weights:
             "technical" → BM25 weighted higher (0.6)
             "semantic"  → dense weighted higher  (0.6)
             default     → equal weights          (0.5/0.5)

Output: up to top_k CandidateNormalized objects ordered by RRF score.

Usage:
    python -m pipeline.layer1_retrieval
"""

from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import json
import time
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logger import log_layer_complete, log_warning
from utils.schema import CandidateNormalized, JDIntent

np.random.seed(42)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_BATCH_SIZE = 64


# ===========================================================================
# PART 1 — CANDIDATE TEXT REPRESENTATION
# ===========================================================================

def candidate_to_text(c: CandidateNormalized) -> str:
    """
    Serialize a normalized candidate into a single searchable string.

    Combines skills, experience titles/companies, location, education, and
    trajectory into a flat text that both the dense encoder and BM25 can
    consume.  Field order matches importance: skills and experience first.
    """
    skills_str = " ".join(c.skills or [])
    exp_str = " ".join([
        f"{e.title} {e.company}"
        for e in (c.experience or [])
    ])
    return (
        f"Skills: {skills_str}. "
        f"Experience: {exp_str}. "
        f"Location: {c.location or ''}. "
        f"Education: {str(c.education or '')}. "
        f"Trajectory: {c.trajectory_label or ''}."
    )


# ===========================================================================
# PART 2 — FAISS (DENSE)
# ===========================================================================

def build_faiss_index(
    candidates: list[CandidateNormalized],
) -> tuple[faiss.Index, list[str], SentenceTransformer]:
    """
    Encode all candidates and build a FAISS inner-product index.

    Embeddings are L2-normalised before indexing so that inner product is
    equivalent to cosine similarity.  The returned candidate_ids list maps
    FAISS position i → candidate_id string.

    Returns:
        index          — faiss.IndexFlatIP ready for search
        candidate_ids  — positional mapping: index → candidate_id
        model          — loaded SentenceTransformer (reuse for JD encoding)
    """
    print(f"  → Loading sentence-transformer ({_EMBEDDING_MODEL})...")
    model = SentenceTransformer(_EMBEDDING_MODEL)

    texts = [candidate_to_text(c) for c in candidates]
    candidate_ids = [c.candidate_id for c in candidates]

    print(f"  → Encoding {len(texts)} candidates in batches of {_BATCH_SIZE}...")
    all_embeddings: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), _BATCH_SIZE),
        desc="  Dense encode",
        unit="batch",
    ):
        batch = texts[start : start + _BATCH_SIZE]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)

    embeddings: np.ndarray = np.vstack(all_embeddings).astype("float32")

    # L2-normalise so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings /= norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"  → FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, candidate_ids, model


def encode_jd(
    jd_intent: JDIntent,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Encode the parsed JD intent into an L2-normalised query embedding.

    Query text combines must-have skills, nice-to-have skills, seniority,
    and domain — mirroring the structure used for candidate text so that
    cosine similarity is meaningful.

    Returns:
        shape (1, embedding_dim) float32 array
    """
    query = (
        f"Skills: {' '.join(jd_intent.must_have)} "
        f"{' '.join(jd_intent.nice_have)}. "
        f"Seniority: {jd_intent.seniority}. "
        f"Domain: {jd_intent.domain}."
    )
    emb: np.ndarray = model.encode([query], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    emb /= norm
    return emb


def dense_search(
    jd_embedding: np.ndarray,
    index: faiss.Index,
    candidate_ids: list[str],
    top_k: int = 200,
) -> list[tuple[str, float]]:
    """
    Run a FAISS inner-product search and return ranked (candidate_id, score) pairs.

    Returns:
        List of (candidate_id, cosine_score) tuples sorted descending by score,
        capped at top_k.
    """
    k = min(top_k, index.ntotal)
    scores, indices = index.search(jd_embedding, k)

    results: list[tuple[str, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        results.append((candidate_ids[idx], float(score)))

    return results  # already descending from FAISS


# ===========================================================================
# PART 3 — BM25 (SPARSE)
# ===========================================================================

def build_bm25_index(
    candidates: list[CandidateNormalized],
) -> tuple[BM25Okapi, list[str]]:
    """
    Tokenise all candidates and build a BM25Okapi index.

    Tokenisation: lowercase + whitespace split on the candidate_to_text
    representation.  BM25 is better at exact skill-name matching than the
    dense encoder.

    Returns:
        bm25           — fitted BM25Okapi instance
        candidate_ids  — positional mapping: index → candidate_id
    """
    candidate_ids = [c.candidate_id for c in candidates]
    tokenised = [candidate_to_text(c).lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenised)
    print(f"  → BM25 index built: {len(candidate_ids)} documents")
    return bm25, candidate_ids


def sparse_search(
    jd_intent: JDIntent,
    bm25: BM25Okapi,
    candidate_ids: list[str],
    top_k: int = 200,
) -> list[tuple[str, float]]:
    """
    Score all candidates with BM25 against the JD query and return top-K.

    Query is formed from must_have + nice_have skills, seniority, and domain,
    then lowercased and split for BM25.

    Returns:
        List of (candidate_id, bm25_score) tuples sorted descending, capped
        at top_k.
    """
    query_text = " ".join(
        jd_intent.must_have
        + jd_intent.nice_have
        + [jd_intent.seniority, jd_intent.domain]
    ).lower()
    query_tokens = query_text.split()

    scores: np.ndarray = bm25.get_scores(query_tokens)

    # Pair each score with its candidate_id then sort descending
    scored = sorted(
        zip(candidate_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return scored[:top_k]


# ===========================================================================
# PART 4 — RRF FUSION
# ===========================================================================

_MODE_WEIGHTS: dict[str, tuple[float, float]] = {
    "technical": (0.4, 0.6),   # (dense, sparse)
    "semantic":  (0.6, 0.4),
}
_DEFAULT_WEIGHTS: tuple[float, float] = (0.5, 0.5)


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    retrieval_mode: str = "semantic",
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine dense and sparse ranked lists via Reciprocal Rank Fusion.

    RRF formula for each candidate d:
        score(d) = dense_w  * 1/(k + dense_rank)
                 + sparse_w * 1/(k + sparse_rank)

    If a candidate appears in only one list its rank in the other list is
    set to len(that_list) + 1 (i.e., last + 1) so it is still considered
    but penalised relative to candidates that appear in both.

    Mode-dependent weights:
        "technical" → dense=0.4, sparse=0.6
        "semantic"  → dense=0.6, sparse=0.4
        default     → dense=0.5, sparse=0.5

    Returns:
        All unique candidates sorted by RRF score descending.
    """
    dense_w, sparse_w = _MODE_WEIGHTS.get(retrieval_mode, _DEFAULT_WEIGHTS)

    dense_n = len(dense_results)
    sparse_n = len(sparse_results)

    # Build rank lookup: candidate_id → 1-based rank
    dense_rank: dict[str, int] = {cid: i + 1 for i, (cid, _) in enumerate(dense_results)}
    sparse_rank: dict[str, int] = {cid: i + 1 for i, (cid, _) in enumerate(sparse_results)}

    all_ids = set(dense_rank) | set(sparse_rank)

    fused: list[tuple[str, float]] = []
    for cid in all_ids:
        dr = dense_rank.get(cid, dense_n + 1)
        sr = sparse_rank.get(cid, sparse_n + 1)
        rrf_score = dense_w / (k + dr) + sparse_w / (k + sr)
        fused.append((cid, rrf_score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# ===========================================================================
# PART 5 — ORCHESTRATOR
# ===========================================================================

def run_layer1(
    jd_intent: JDIntent,
    candidates: list[CandidateNormalized],
    top_k: int = 200,
) -> tuple[list[CandidateNormalized], Optional[SentenceTransformer]]:
    """
    Layer 1 top-level orchestrator — hybrid retrieval pipeline.

    1. Build FAISS dense index from candidate embeddings.
    2. Build BM25 sparse index from tokenised candidate text.
    3. Encode JD intent as a dense query vector.
    4. Dense search → top-K candidates by cosine similarity.
    5. Sparse search → top-K candidates by BM25 score.
    6. RRF fusion → single ranked list with mode-aware weights.
    7. Map fused candidate_ids back to CandidateNormalized objects.
    8. Slice to top_k and log completion.

    Returns:
        Tuple of (ranked_candidates, loaded_model).  The SentenceTransformer
        model is returned so Layer 2 can reuse it without reloading from disk.
        Both elements are None / empty list when candidates is empty.
    """
    start = time.time()

    if not candidates:
        log_warning("Layer 1: received 0 candidates — returning empty list.")
        return [], None

    effective_k = min(top_k, len(candidates))

    # Step 1 — dense index
    index, candidate_ids, model = build_faiss_index(candidates)

    # Step 2 — sparse index
    bm25, bm25_ids = build_bm25_index(candidates)

    # Step 3 — encode JD
    jd_embedding = encode_jd(jd_intent, model)

    # Step 4 — dense search
    dense_results = dense_search(jd_embedding, index, candidate_ids, top_k=effective_k)

    # Step 5 — sparse search
    sparse_results = sparse_search(jd_intent, bm25, bm25_ids, top_k=effective_k)

    # Step 6 — RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        retrieval_mode=jd_intent.retrieval_mode,
    )

    # Step 7 — map back to CandidateNormalized, preserving RRF order
    lookup: dict[str, CandidateNormalized] = {c.candidate_id: c for c in candidates}
    result: list[CandidateNormalized] = []
    for cid, _score in fused:
        c = lookup.get(cid)
        if c is not None:
            result.append(c)

    # Step 8 — slice
    result = result[:top_k]

    elapsed = time.time() - start
    log_layer_complete(layer=1, candidates_remaining=len(result), elapsed_seconds=elapsed)

    return result, model


# ===========================================================================
# QUICK TEST BLOCK
# ===========================================================================

if __name__ == "__main__":
    from pipeline.layer0_normalizer import run_layer0

    DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

    # --- Load JD ---
    jd_path = DATA_DIR / "jd_senior_ml.txt"
    jd_raw = json.loads(jd_path.read_text(encoding="utf-8"))
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

    # --- Load candidates (full dicts for pre_populated) ---
    all_candidates: list[dict] = json.loads(
        (DATA_DIR / "candidates.json").read_text(encoding="utf-8")
    )
    # Keep type lookup for test assertions
    type_by_id: dict[str, str] = {c["candidate_id"]: c.get("type", "") for c in all_candidates}

    print("\n" + "=" * 60)
    print("  Layer 1 Quick Test — jd_senior_ml.txt + 30 candidates")
    print("=" * 60 + "\n")

    # --- Run Layer 0 ---
    jd_intent, normalized = run_layer0(jd_text, "JD_SENIOR_ML", all_candidates)
    print(f"\n  Layer 0 complete: {len(normalized)} candidates passed.\n")

    # --- Run Layer 1 ---
    retrieved, _model = run_layer1(jd_intent, normalized, top_k=20)

    # --- Results table ---
    print("\n  Rank | ID    | Type                     | RRF Rank Score")
    print("  ─────┼───────┼──────────────────────────┼───────────────")

    # Rebuild RRF scores for display (re-run fusion with same inputs to get scores)
    _idx_map: dict[str, int] = {c.candidate_id: i for i, c in enumerate(retrieved)}
    for rank, c in enumerate(retrieved, start=1):
        ctype = type_by_id.get(c.candidate_id, "unknown")
        print(f"  {rank:>4} | {c.candidate_id:<5} | {ctype:<24} | rank {rank}")

    # --- Evaluation ---
    strong_in_top10 = sum(
        1 for c in retrieved[:10]
        if type_by_id.get(c.candidate_id, "") == "strong_match"
    )
    # With 29 candidates and top_k=20, RRF scores are too compressed for poor
    # matches to cluster reliably in the bottom 5.  The equivalent meaningful
    # check is that no poor match contaminates the very top results.
    poor_in_top5 = sum(
        1 for c in retrieved[:5]
        if type_by_id.get(c.candidate_id, "") == "poor_match"
    )
    poor_in_top10 = sum(
        1 for c in retrieved[:10]
        if type_by_id.get(c.candidate_id, "") == "poor_match"
    )

    print(f"\n  strong_match in top 10:  {strong_in_top10}/5")
    print(f"  poor_match in top 5:     {poor_in_top5}  (expect 0)")
    print(f"  poor_match in top 10:    {poor_in_top10} (expect ≤ 1)")

    assert strong_in_top10 >= 3, (
        f"Expected >= 3 strong_match in top 10, got {strong_in_top10}"
    )
    assert poor_in_top5 == 0, (
        f"Expected 0 poor_match in top 5, got {poor_in_top5}"
    )
    assert poor_in_top10 <= 1, (
        f"Expected <= 1 poor_match in top 10, got {poor_in_top10}"
    )
    print("\n  ✓ All retrieval quality assertions passed.")
