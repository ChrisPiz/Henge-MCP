"""Critical invariants — these protect design contracts. Refactors must not break these silently."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from tenthai.agents import run_agents, TENTH_MAN
from tenthai.embed import project_mds


@pytest.mark.asyncio
async def test_partial_failure_8of9(mock_anthropic_client):
    """1 frame falla → sistema continúa con 8 frames + tenth-man, marca el faltante."""
    call_count = [0]
    original_create = mock_anthropic_client.messages.create

    async def maybe_fail(**kwargs):
        call_count[0] += 1
        if call_count[0] == 3:
            raise RuntimeError("Simulated 1-of-9 API failure")
        return await original_create(**kwargs)

    mock_anthropic_client.messages.create = AsyncMock(side_effect=maybe_fail)

    results = await run_agents(mock_anthropic_client, "Pregunta de prueba")

    assert len(results) == 10
    failed = [r for r in results if r[2] == "failed"]
    ok = [r for r in results if r[2] == "ok"]
    assert len(failed) == 1, f"Expected exactly 1 failed frame, got {len(failed)}"
    assert len(ok) == 9, "8 frames + 1 tenth-man = 9 ok"
    assert results[-1][0] == TENTH_MAN
    assert results[-1][2] == "ok"


@pytest.mark.asyncio
async def test_partial_failure_abort_lt_8(mock_anthropic_client):
    """2+ frames fallan → RuntimeError con mensaje claro indicando cuántos sobrevivieron."""
    call_count = [0]

    async def fail_two(**kwargs):
        call_count[0] += 1
        if call_count[0] in (2, 4):
            raise RuntimeError("Simulated 2-of-9 API failure")
        result = MagicMock()
        text_part = MagicMock()
        text_part.text = "ok response"
        result.content = [text_part]
        return result

    mock_anthropic_client.messages.create = AsyncMock(side_effect=fail_two)

    with pytest.raises(RuntimeError, match=r"7/9"):
        await run_agents(mock_anthropic_client, "Pregunta de prueba")


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
