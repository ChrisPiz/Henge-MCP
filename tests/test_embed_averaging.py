"""Phase 7: default model bump + EMBED_AVERAGE mode."""
import pytest
import numpy as np
from unittest.mock import patch

from henge.embed import (
    _resolve_provider,
    embed_responses,
    project_mds,
)


def test_default_openai_model_is_3_large():
    provider, model, _fn = _resolve_provider()
    assert provider == "openai"
    assert model == "text-embedding-3-large"


def test_resolve_provider_voyage_when_set(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    provider, model, _fn = _resolve_provider()
    assert provider == "voyage"
    assert model == "voyage-3-large"


def test_project_mds_with_averaging_returns_same_shape():
    """When embeddings_extra is provided, project_mds averages the distance matrices."""
    rng = np.random.default_rng(7)
    primary = rng.normal(0, 0.1, size=(10, 64))
    primary = primary / np.linalg.norm(primary, axis=1, keepdims=True)
    extra = rng.normal(0, 0.1, size=(10, 256))  # different dim, valid since each set has its own distance matrix
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)

    result = project_mds(primary.tolist(), embeddings_extra=extra.tolist())

    assert "coords_2d" in result
    assert "distance_to_centroid_of_9" in result
    assert len(result["distance_to_centroid_of_9"]) == 10
    assert len(result["coords_2d"]) == 10
    assert result["n_frames"] == 9


def test_project_mds_averaging_differs_from_single():
    """Averaging two distinct distance matrices gives a different MDS layout than either alone."""
    rng = np.random.default_rng(11)
    a = rng.normal(0, 0.1, size=(10, 64))
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(0, 0.1, size=(10, 64))
    b = b / np.linalg.norm(b, axis=1, keepdims=True)

    single = project_mds(a.tolist())
    averaged = project_mds(a.tolist(), embeddings_extra=b.tolist())

    # MDS coordinates should differ because the averaged distance matrix injects
    # signal from the second embedding set. The centroid distances stay in the
    # primary embedding space (by design), so we compare 2D coords instead.
    single_coords = np.array(single["coords_2d"])
    averaged_coords = np.array(averaged["coords_2d"])

    # Compute the Frobenius norm of the difference (after aligning sign/rotation
    # is not needed since we just need any difference).
    # Use pairwise distance matrices to be rotation-invariant:
    from scipy.spatial.distance import pdist
    d_single = pdist(single_coords)
    d_averaged = pdist(averaged_coords)
    max_diff = float(np.max(np.abs(d_single - d_averaged)))
    assert max_diff > 1e-3, "averaging should shift the MDS layout"


def test_embed_responses_average_off_by_default():
    """Default behavior unchanged: single provider, no embeddings_extra."""
    captured = []

    def fake_openai(texts, model=None):
        captured.append(("openai", model))
        return [[0.1] * 64 for _ in texts]

    # Patch the cache to always miss so the fake embed fn is actually called.
    with patch("henge.embed._embed_openai", side_effect=fake_openai), \
         patch("henge.embed._cached_embedding", return_value=None):
        result = embed_responses(["a", "b", "c"])

    assert result["ok"]
    assert result["provider"] == "openai"
    assert result["model"] == "text-embedding-3-large"
    assert "embeddings_extra" not in result or result.get("embeddings_extra") is None
    # Only one provider call:
    assert any(c[0] == "openai" for c in captured)


def test_embed_responses_average_on_calls_both(monkeypatch):
    """When EMBED_AVERAGE=true, both OpenAI and Voyage are called and embeddings_extra is populated."""
    monkeypatch.setenv("EMBED_AVERAGE", "true")
    # Force a fresh import to pick up the flag at call time, OR check that
    # the runtime path reads the env var at each call.

    captured = []

    def fake_openai(texts, model=None):
        captured.append(("openai", model))
        return [[0.1] * 64 for _ in texts]

    def fake_voyage(texts, model=None):
        captured.append(("voyage", model))
        return [[0.2] * 1024 for _ in texts]

    with patch("henge.embed._embed_openai", side_effect=fake_openai), \
         patch("henge.embed._embed_voyage", side_effect=fake_voyage), \
         patch("henge.embed._cached_embedding", return_value=None):
        # Bypass the cache so both fakes are actually called.
        result = embed_responses([f"unique-text-{i}" for i in range(3)])

    assert result["ok"]
    # Either one of them might fail in env without the key — accept that gracefully.
    # In the test env, both fakes are patched in-process, so both should fire.
    providers_called = {c[0] for c in captured}
    assert "openai" in providers_called
    assert "voyage" in providers_called
    # When both succeed, embeddings_extra is populated:
    assert result.get("embeddings_extra") is not None
    assert len(result["embeddings_extra"]) == 3


def test_embed_responses_average_falls_back_when_voyage_fails(monkeypatch):
    """If Voyage fails under EMBED_AVERAGE, return OpenAI-only result without crash."""
    monkeypatch.setenv("EMBED_AVERAGE", "true")

    def fake_openai(texts, model=None):
        return [[0.1] * 64 for _ in texts]

    def fake_voyage(texts, model=None):
        raise RuntimeError("Voyage not available")

    with patch("henge.embed._embed_openai", side_effect=fake_openai), \
         patch("henge.embed._embed_voyage", side_effect=fake_voyage), \
         patch("henge.embed._cached_embedding", return_value=None):
        result = embed_responses([f"unique-text-fallback-{i}" for i in range(3)])

    assert result["ok"]
    assert result.get("embeddings_extra") is None
    assert result.get("average_partial") is True or "embeddings_extra" not in result
