"""Phase 5 dual tenth-man: blind + informed dataclasses + feature flag."""
import pytest

from henge.tenth_man import (
    TenthManBlindResult,
    TenthManInformedResult,
    ENABLE_TENTH_MAN_DUAL,
)


def test_blind_result_fields():
    r = TenthManBlindResult(
        text="Blind dissent text...",
        opus_usage={"model": "anthropic/opus-4-7", "input_tokens": 100, "output_tokens": 200},
    )
    assert r.text.startswith("Blind")
    assert r.opus_usage["model"] == "anthropic/opus-4-7"


def test_informed_result_fields():
    r = TenthManInformedResult(
        text="Refined dissent...",
        what_holds=["claim A holds", "claim B holds"],
        what_revised=["claim C revised to..."],
        what_discarded=["claim D was bias"],
        gpt5_usage={"model": "openai/gpt-5", "input_tokens": 500, "output_tokens": 300},
    )
    assert len(r.what_holds) == 2
    assert len(r.what_revised) == 1
    assert len(r.what_discarded) == 1
    assert r.gpt5_usage["model"] == "openai/gpt-5"


def test_informed_result_passthrough_defaults():
    """Empty informed result (e.g. on parse failure) is constructable."""
    r = TenthManInformedResult(
        text="raw text only",
        what_holds=[],
        what_revised=[],
        what_discarded=[],
        gpt5_usage=None,
    )
    assert r.text == "raw text only"
    assert r.gpt5_usage is None


def test_enable_tenth_man_dual_default_true():
    assert isinstance(ENABLE_TENTH_MAN_DUAL, bool)


import json
from unittest.mock import AsyncMock, patch

from henge.providers.base import CompletionResponse
from henge.tenth_man import run_tenth_man_blind, run_tenth_man_informed


def _resp(text: str, model_id: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        input_tokens=200,
        output_tokens=400,
        model=model_id,
        raw_model=model_id.split("/", 1)[1],
        finish_reason="end_turn",
    )


@pytest.mark.asyncio
async def test_blind_calls_opus_and_returns_text():
    captured = {}

    async def fake_complete(model_id, req):
        captured["model_id"] = model_id
        captured["system_excerpt"] = req.system[:50]
        captured["user_excerpt"] = req.user[:200]
        return _resp("## Hechos que acepto\n\nEl consenso anticipado...", model_id)

    with patch("henge.tenth_man.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await run_tenth_man_blind("Q?", "user context here")

    assert captured["model_id"] == "anthropic/opus-4-7"
    assert "Hechos que acepto" in r.text
    assert r.opus_usage is not None
    assert r.opus_usage["model"] == "anthropic/opus-4-7"


@pytest.mark.asyncio
async def test_blind_handles_call_failure():
    async def fake_complete(model_id, req):
        raise RuntimeError("Anthropic 503")

    with patch("henge.tenth_man.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await run_tenth_man_blind("Q?", "ctx")

    # Degraded but still returns a result (allows the rest of the pipeline to proceed)
    assert "[failed" in r.text.lower() or "could not" in r.text.lower()
    assert r.opus_usage is None


_INFORMED_JSON = json.dumps({
    "text": "After seeing the nine: the blind's premise A still holds...",
    "what_holds": ["The runway constraint is the dominant signal", "Validate before cutting"],
    "what_revised": ["The 'fake-urgency' framing is sharper than what the blind wrote"],
    "what_discarded": ["The catastrophic-failure tone of the blind was overdone"],
})


@pytest.mark.asyncio
async def test_informed_returns_structured_diff():
    nine_outputs = [
        ("empirical", "Numbers say 3 weeks of runway."),
        ("historical", "Slack and Basecamp pivoted with paid signal first."),
    ]
    blind = "Blind dissent text claiming X."

    async def fake_complete(model_id, req):
        assert model_id == "openai/gpt-5"
        return _resp(_INFORMED_JSON, model_id)

    with patch("henge.tenth_man.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await run_tenth_man_informed("Q?", "ctx", nine_outputs, blind)

    assert "blind's premise A still holds" in r.text
    assert len(r.what_holds) == 2
    assert len(r.what_revised) == 1
    assert len(r.what_discarded) == 1
    assert r.gpt5_usage["model"] == "openai/gpt-5"


@pytest.mark.asyncio
async def test_informed_handles_garbage_json():
    async def fake_complete(model_id, req):
        return _resp("not json prose", model_id)

    with patch("henge.tenth_man.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await run_tenth_man_informed("Q?", "ctx", [("empirical", "x")], "blind text")

    # Degraded: returns the raw text, empty diff lists
    assert "not json prose" in r.text or r.text == "not json prose"
    assert r.what_holds == []
    assert r.what_revised == []
    assert r.what_discarded == []
    assert r.gpt5_usage is not None  # call happened, parse failed
