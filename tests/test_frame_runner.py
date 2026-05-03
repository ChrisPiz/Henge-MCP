"""Each frame must call providers.complete with its assigned canonical_id from FRAME_MODEL_MAP."""
import pytest
from unittest.mock import AsyncMock, patch

from henge.agents import run_agent
from henge.config.frame_assignment import FRAME_MODEL_MAP
from henge.providers.base import CompletionResponse


def _fake_response(model_id: str) -> CompletionResponse:
    return CompletionResponse(
        text=f"response from {model_id}",
        input_tokens=100,
        output_tokens=50,
        model=model_id,
        raw_model=model_id.split("/", 1)[1],
        finish_reason="end_turn",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("frame,expected_model", list(FRAME_MODEL_MAP.items()))
async def test_run_agent_dispatches_to_assigned_model(frame, expected_model):
    captured: dict = {}

    async def fake_complete(model_id, req):
        captured["model_id"] = model_id
        captured["req"] = req
        return _fake_response(model_id)

    with patch("henge.agents.complete", new=AsyncMock(side_effect=fake_complete)):
        text, usage = await run_agent(frame, "Pregunta de prueba")

    assert captured["model_id"] == expected_model
    assert text.startswith("response from")
    assert usage["model"] == expected_model
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


@pytest.mark.asyncio
async def test_run_agent_unknown_frame_raises():
    with pytest.raises(KeyError):
        await run_agent("not-a-frame", "Pregunta")
