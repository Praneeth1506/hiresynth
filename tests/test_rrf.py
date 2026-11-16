"""
Tests for Reciprocal Rank Fusion (pipeline/layer1_retrieval.py).

RRF combines the dense (FAISS) and sparse (BM25) ranked lists:
    score(d) = dense_w / (k + dense_rank) + sparse_w / (k + sparse_rank)
with mode weights technical=(0.4, 0.6), semantic=(0.6, 0.4), other=(0.5, 0.5)
and a missing candidate assigned rank len(list) + 1 in that list.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.layer1_retrieval import reciprocal_rank_fusion

REPO_ROOT = Path(__file__).resolve().parent.parent


def ids(fused):
    """Strip scores from an RRF result."""
    return [cid for cid, _ in fused]


class TestFormula:
    """Exact scores for small hand-checked inputs."""

    def test_exact_scores_technical_mode(self):
        dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        sparse = [("C", 12.0), ("A", 10.0)]  # B missing → sparse rank 3
        fused = dict(reciprocal_rank_fusion(dense, sparse, "technical", k=60))
        assert fused["A"] == pytest.approx(0.4 / 61 + 0.6 / 62)
        assert fused["B"] == pytest.approx(0.4 / 62 + 0.6 / 63)
        assert fused["C"] == pytest.approx(0.4 / 63 + 0.6 / 61)

    def test_sorted_descending(self):
        dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        sparse = [("C", 12.0), ("A", 10.0)]
        fused = reciprocal_rank_fusion(dense, sparse, "technical")
        assert ids(fused) == ["A", "C", "B"]
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_input_scores_are_ignored_only_ranks_matter(self):
        dense_a = [("A", 0.99), ("B", 0.01)]
        dense_b = [("A", 5.0), ("B", 4.9)]
        sparse = [("B", 1.0), ("A", 0.5)]
        assert reciprocal_rank_fusion(dense_a, sparse, "semantic") == (
            reciprocal_rank_fusion(dense_b, sparse, "semantic")
        )

    def test_k_controls_rank_sensitivity(self):
        dense = [("A", 1.0), ("B", 0.5)]
        sparse = [("A", 1.0), ("B", 0.5)]
        small_k = dict(reciprocal_rank_fusion(dense, sparse, k=1))
        large_k = dict(reciprocal_rank_fusion(dense, sparse, k=1000))
        assert small_k["A"] / small_k["B"] > large_k["A"] / large_k["B"]


class TestModeWeights:
    """retrieval_mode decides which retriever wins a disagreement."""

    DENSE = [("DENSE_FAV", 0.9), ("SPARSE_FAV", 0.8)]
    SPARSE = [("SPARSE_FAV", 9.0), ("DENSE_FAV", 8.0)]

    def test_technical_favours_bm25(self):
        fused = reciprocal_rank_fusion(self.DENSE, self.SPARSE, "technical")
        assert ids(fused) == ["SPARSE_FAV", "DENSE_FAV"]

    def test_semantic_favours_dense(self):
        fused = reciprocal_rank_fusion(self.DENSE, self.SPARSE, "semantic")
        assert ids(fused) == ["DENSE_FAV", "SPARSE_FAV"]

    def test_unknown_mode_uses_equal_weights(self):
        fused = dict(reciprocal_rank_fusion(self.DENSE, self.SPARSE, "something_else"))
        assert fused["DENSE_FAV"] == pytest.approx(0.5 / 61 + 0.5 / 62)
        assert fused["DENSE_FAV"] == pytest.approx(fused["SPARSE_FAV"])


class TestMembership:
    """Union semantics and handling of candidates found by only one retriever."""

    def test_returns_union_without_duplicates(self):
        dense = [("A", 1.0), ("B", 0.9)]
        sparse = [("C", 3.0), ("A", 2.0)]
        fused = reciprocal_rank_fusion(dense, sparse)
        assert sorted(ids(fused)) == ["A", "B", "C"]

    def test_found_by_both_beats_found_by_one_at_same_rank(self):
        dense = [("BOTH", 1.0), ("DENSE_ONLY", 0.9)]
        sparse = [("SPARSE_ONLY", 2.0), ("BOTH", 1.0)]
        fused = reciprocal_rank_fusion(dense, sparse, "semantic")
        assert ids(fused)[0] == "BOTH"

    def test_one_list_empty(self):
        dense = [("A", 1.0), ("B", 0.5)]
        fused = reciprocal_rank_fusion(dense, [], "semantic")
        assert ids(fused) == ["A", "B"]
        # Missing from sparse → sparse rank = 0 + 1
        assert dict(fused)["A"] == pytest.approx(0.6 / 61 + 0.4 / 61)

    def test_both_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_agreeing_lists_preserve_order(self):
        order = [f"C{i:03d}" for i in range(20)]
        dense = [(c, 1.0 - i / 100) for i, c in enumerate(order)]
        sparse = [(c, 50.0 - i) for i, c in enumerate(order)]
        for mode in ("technical", "semantic", "other"):
            assert ids(reciprocal_rank_fusion(dense, sparse, mode)) == order


class TestDeterminism:
    """Ties must be broken the same way in every process."""

    # Mirrored ranks under equal weights give exact ties: C001/C002 and C003/C004.
    DENSE = [("C001", 0.9), ("C002", 0.8), ("C003", 0.7), ("C004", 0.6)]
    SPARSE = [("C002", 9.0), ("C001", 8.0), ("C004", 7.0), ("C003", 6.0)]

    def test_ties_broken_by_candidate_id(self):
        fused = reciprocal_rank_fusion(self.DENSE, self.SPARSE, "other")
        assert ids(fused) == ["C001", "C002", "C003", "C004"]

    def test_same_result_across_hash_seeds(self):
        """
        Candidate IDs are collected in a set; string hashing is randomised per
        process, so without an explicit tie-break the order of tied candidates
        would change from run to run.
        """
        script = (
            "import json\n"
            "from pipeline.layer1_retrieval import reciprocal_rank_fusion\n"
            f"d = {self.DENSE!r}\n"
            f"s = {self.SPARSE!r}\n"
            "print(json.dumps([c for c, _ in reciprocal_rank_fusion(d, s, 'other')]))\n"
        )
        outputs = set()
        for seed in ("0", "1", "2", "3"):  # each subprocess imports torch, keep it small
            env = {**os.environ, "PYTHONHASHSEED": seed}
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
            )
            outputs.add(tuple(json.loads(result.stdout.strip().splitlines()[-1])))
        assert outputs == {("C001", "C002", "C003", "C004")}
