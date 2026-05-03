"""Critical invariants — these protect design contracts. Refactors must not break these silently."""
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from henge.agents import PROMPTS, PROMPTS_HASH, TEMPERATURE, TENTH_MAN, run_agents
from henge.embed import project_mds
from henge.providers.pricing import (
    PRICING_VERSION,
    build_cost_breakdown,
    cost_for,
)
from henge.viz import compute_cfi


@pytest.mark.asyncio
async def test_partial_failure_8of9(mock_anthropic_client, mock_providers):
    """1 frame falla → sistema continúa con 8 frames + blind + informed, marca el faltante.

    The blind tenth-man fires in parallel with the 9; informed runs sequentially.
    Failure injection on the 3rd frame call only — blind path is untouched.
    """
    call_count = [0]
    base_side_effect = mock_providers.side_effect

    async def maybe_fail(model_id, req):
        call_count[0] += 1
        # Skip the blind/informed paths — only inject on the 3rd FRAME call.
        if model_id in ("anthropic/opus-4-7", "openai/gpt-5") and (
            "tenth" in (req.system or "").lower() or "blind dissent" in (req.system or "").lower()
        ):
            return await base_side_effect(model_id, req)
        if call_count[0] == 3:
            raise RuntimeError("Simulated 1-of-9 frame failure")
        return await base_side_effect(model_id, req)

    mock_providers.side_effect = maybe_fail

    result = await run_agents("Pregunta de prueba")

    assert isinstance(result, dict)
    nine = result["nine"]
    assert len(nine) == 9
    failed = [r for r in nine if r[2] == "failed"]
    assert len(failed) == 1, f"Expected exactly 1 failed frame, got {len(failed)}"
    assert "blind" in result
    assert "informed" in result


@pytest.mark.asyncio
async def test_partial_failure_abort_lt_8(mock_anthropic_client, mock_providers):
    """2+ frames fallan → RuntimeError con mensaje claro indicando cuántos sobrevivieron."""
    call_count = [0]
    base_side_effect = mock_providers.side_effect

    async def fail_two(model_id, req):
        call_count[0] += 1
        if call_count[0] in (2, 4):
            raise RuntimeError("Simulated 2-of-9 frame failure")
        return await base_side_effect(model_id, req)

    mock_providers.side_effect = fail_two

    with pytest.raises(RuntimeError, match=r"7/9"):
        await run_agents("Pregunta de prueba")


def test_centroid_excludes_tenth(synthetic_embeddings_10):
    """centroid_of_9 debe computarse SOLO sobre los primeros 9, excluyendo el #10.

    Synthetic setup: 9 puntos clusterizan en una dirección, 1 outlier está lejos.
    Si el centroide incluye al #10, las distancias se diluyen y este test falla.
    """
    proj = project_mds(synthetic_embeddings_10)
    distances = proj["distance_to_centroid_of_9"]

    assert len(distances) == 10
    max_frame_dist = max(distances[:9])
    tenth_dist = distances[9]
    assert tenth_dist > max_frame_dist, (
        f"tenth-man distance ({tenth_dist:.3f}) debe exceder max frame distance "
        f"({max_frame_dist:.3f}). Si no, el centroide está contaminado por #10."
    )


# ──────────────────────────────────────────────────────────────────────────
# v0.5 — Reproducibility kit + embed-fix regression tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_temperature_is_zero(mock_anthropic_client, mock_providers):
    """Cada call a través del registry pasa temperature=0 por reproducibilidad.

    Blind (Opus) tolera temperature=0; Anthropic provider lo silencia para Opus.
    Frames + informed pasan temperature explícito al CompletionRequest.
    """
    captured = []
    base_side_effect = mock_providers.side_effect

    async def capture(model_id, req):
        captured.append((model_id, req.temperature))
        return await base_side_effect(model_id, req)

    mock_providers.side_effect = capture

    await run_agents("Pregunta de prueba")

    assert TEMPERATURE == 0, "Module-level TEMPERATURE constant must be 0"

    # 9 frames + 1 blind + 1 informed = 11 calls total
    assert len(captured) == 11, f"Expected 11 calls, got {len(captured)}"
    for model_id, temp in captured:
        assert temp == 0, f"{model_id} should pass temperature=0; got {temp!r}"


def test_project_mds_excludes_failed_frames():
    """project_mds(..., n_frames=8) debe centrar el centroide en 8 frames, no 9.

    Esto verifica el fix v0.5 al bug donde texts incluía stubs de frames
    fallidos al embedding. Con n_frames=8 + tenth, el centroide ignora
    cualquier frame fallido (que ya no se embebe).
    """
    import numpy as np

    rng = np.random.default_rng(7)
    # 8 frames cluster around +x, the tenth (last) is opposite
    cluster = rng.normal(0, 0.1, size=(8, 256))
    cluster[:, 0] += 1.0
    cluster = cluster / np.linalg.norm(cluster, axis=1, keepdims=True)
    tenth = rng.normal(0, 0.1, size=(1, 256))
    tenth[:, 0] -= 1.0
    tenth = tenth / np.linalg.norm(tenth, axis=1, keepdims=True)
    embeddings = np.vstack([cluster, tenth]).tolist()

    proj = project_mds(embeddings, n_frames=8)
    assert proj["n_frames"] == 8
    assert len(proj["distance_to_centroid_of_9"]) == 9  # 8 frames + tenth
    # Tenth must be furthest because the 8 cluster on one side
    distances = proj["distance_to_centroid_of_9"]
    assert distances[8] > max(distances[:8]), (
        "Tenth-man must be further from the centroid than any successful frame"
    )


def test_prompts_hash_stable():
    """``PROMPTS_HASH`` es determinístico — mismo prompt set → mismo hash.

    Persistido en cada report. Cualquier cambio a los .md cambia el hash;
    reports con hashes distintos no son directamente comparables.
    """
    import hashlib

    from henge.agents import FRAMES

    blob = "".join(PROMPTS[name] for name in [*FRAMES, TENTH_MAN]).encode("utf-8")
    expected = hashlib.sha256(blob).hexdigest()[:16]
    assert PROMPTS_HASH == expected
    assert len(PROMPTS_HASH) == 16
    # SHA256 hex prefix → only [0-9a-f]
    assert all(c in "0123456789abcdef" for c in PROMPTS_HASH)


def test_cost_breakdown_sums_components():
    """``build_cost_breakdown`` retorna la suma exacta por bucket.

    Regression contra v0.4 cost_usd hardcoded. v0.6 cost_breakdown debe
    derivarse de uso real (canonical ids), no de un literal. Legacy raw
    SDK strings (e.g. ``claude-haiku-4-5-20251001``) deben ser
    canonicalizados internamente.
    """
    advisor_usages = [
        {"model": "anthropic/sonnet-4-6", "input_tokens": 1000, "output_tokens": 500},
        {"model": "anthropic/sonnet-4-6", "input_tokens": 1200, "output_tokens": 600},
        {"model": "anthropic/opus-4-7",   "input_tokens": 5000, "output_tokens": 2500},
    ]
    scoping = {"model": "anthropic/haiku-4-5", "input_tokens": 200, "output_tokens": 150}
    # Legacy raw SDK string — should be canonicalized internally.
    consensus = {"model": "claude-haiku-4-5-20251001", "input_tokens": 800, "output_tokens": 400}

    expected_anthropic = (
        cost_for("anthropic/sonnet-4-6", 1000, 500)
        + cost_for("anthropic/sonnet-4-6", 1200, 600)
        + cost_for("anthropic/opus-4-7",   5000, 2500)
        + cost_for("anthropic/haiku-4-5", 200, 150)
        + cost_for("anthropic/haiku-4-5", 800, 400)
    )

    breakdown = build_cost_breakdown(
        advisor_usages=advisor_usages,
        blind_usage=None,
        informed_usage=None,
        meta_usage=None,
        canonical_usage=None,
        scoping_haiku_usage=scoping,
        scoping_adversarial_usage=None,
        consensus_usage=consensus,
        claims_extract_usage=None,
        claims_verify_usage=None,
        embedding_model="openai/text-embedding-3-large",
        embedding_input_tokens=2500,
    )

    assert abs(breakdown["anthropic_usd"] - round(expected_anthropic, 6)) < 1e-9
    assert breakdown["openai_usd"] == 0.0  # no openai usages in this fixture
    assert breakdown["embedding_usd"] >= 0
    assert abs(
        breakdown["total_usd"]
        - round(
            breakdown["anthropic_usd"]
            + breakdown["openai_usd"]
            + breakdown["embedding_usd"],
            6,
        )
    ) < 1e-9
    assert breakdown["pricing_version"] == PRICING_VERSION


def test_no_hardcoded_cost_in_logic():
    """Regression: ``cost_usd = 0.65`` ya no debe aparecer como asignación literal.

    En v0.4 el cost vivía hardcoded en server.py. v0.5 lo deriva de usage real.
    Si alguien lo reintroduce, este test falla. Comentarios y docs pueden
    seguir mencionando 0.65 como rango histórico.
    """
    server_path = Path(__file__).parent.parent / "henge" / "server.py"
    text = server_path.read_text(encoding="utf-8")
    # We don't ban the literal 0.65 entirely (docs / comments may reference
    # historical costs). We ban the specific assignment pattern.
    forbidden_patterns = [
        "cost_usd = 0.65",
        'cost_usd= 0.65',
        'cost_usd=0.65',
    ]
    for pat in forbidden_patterns:
        assert pat not in text, (
            f"Regression: {pat!r} found in server.py. v0.5 derives cost from "
            f"actual token usage via henge.providers.pricing.build_cost_breakdown()."
        )


@pytest.mark.asyncio
async def test_k_runs_requires_temperature(monkeypatch):
    """k_runs > 1 sin run_temperature debe fallar con un error claro.

    K-runs es un modo de muestreo de varianza; con temperature=0 todas las
    corridas son idénticas y la distribución es degenerada. El error debe
    decirle al caller que pase run_temperature.
    """
    # Stub the validator so server import doesn't try to ping Anthropic.
    import henge.server as server

    result = await server.decide(
        question="X",
        context="Y",
        k_runs=5,
        run_temperature=None,
    )
    assert result["error"] == "k_runs_requires_temperature"
    assert "run_temperature" in result["reason"]


@pytest.mark.asyncio
async def test_k_runs_requires_context():
    """k_runs > 1 sin context (y sin skip_scoping) debe fallar.

    El modo K-runs salta scoping para no entrar en una recursión interactiva
    durante el muestreo. El caller debe haber recolectado context antes.
    """
    import henge.server as server

    result = await server.decide(
        question="X",
        context=None,
        skip_scoping=False,
        k_runs=3,
        run_temperature=0.7,
    )
    assert result["error"] == "k_runs_requires_context"


def test_compute_cfi_three_bins():
    """CFI clasifica tight clusters según distancia del tenth-man al cluster.

    Pre-registrado en docs/cfi-spec.md:
      - σ ≥ 0.03 → divided
      - else CFI < 0.33 → aligned-stable
      - else → aligned-fragile
    """
    # Tight cluster (σ ≈ 0.012), tenth close → aligned-stable, low CFI
    tight = [0.62, 0.63, 0.64, 0.64, 0.65, 0.65, 0.66, 0.67, 0.68]
    r = compute_cfi(tenth_distance=0.66, frame_distances=tight)
    assert r["cfi_bin"] == "aligned-stable"
    assert 0.0 <= r["cfi"] < 0.33
    assert r["sigma_9"] is not None and r["sigma_9"] < 0.03

    # Tight cluster, tenth pushed far → aligned-fragile, high CFI
    r = compute_cfi(tenth_distance=0.95, frame_distances=tight)
    assert r["cfi_bin"] == "aligned-fragile"
    assert r["cfi"] >= 0.33

    # Spread cluster (σ ≥ 0.03) → divided regardless of tenth
    spread = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.72, 0.78, 0.85]
    r = compute_cfi(tenth_distance=0.90, frame_distances=spread)
    assert r["cfi_bin"] == "divided"
    assert r["sigma_9"] >= 0.03
