"""
Shared pytest fixtures for the HireSynth test suite.

Builders and fakes live in tests/factories.py; this file only wires them
into fixtures that patch module state (clock, cache path).
"""

from __future__ import annotations

import pytest

from tests.factories import FROZEN_NOW, FrozenDateTime


@pytest.fixture
def frozen_layer2_clock(monkeypatch):
    """Freeze datetime.now() inside pipeline.layer2_scoring."""
    import pipeline.layer2_scoring as layer2

    monkeypatch.setattr(layer2, "datetime", FrozenDateTime)
    return FROZEN_NOW


@pytest.fixture
def isolated_llm_cache(tmp_path, monkeypatch):
    """
    Point both cache users (Layer 0 JD cache, Layer 3 LLM cache) at a temp
    file so tests never read or overwrite the real data/llm_cache.json.
    """
    import pipeline.layer0_normalizer as layer0
    import pipeline.layer3_reranker as layer3

    path = tmp_path / "llm_cache.json"
    monkeypatch.setattr(layer0, "_CACHE_PATH", path)
    monkeypatch.setattr(layer3, "_CACHE_PATH", path)
    return path
