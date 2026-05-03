"""Phase 4 meta-frame: MetaFrameResult dataclass + feature flag."""
import pytest

from henge.meta_frame import MetaFrameResult, ENABLE_META_FRAME


def test_meta_frame_result_has_all_fields():
    r = MetaFrameResult(
        decision_class="two-way-with-cost",
        urgency="weeks",
        question_quality="well-formed",
        suggested_reformulation=None,
        meta_recommendation="proceed",
        reasoning="The question presents a clear decision with measurable trade-offs.",
        gpt5_usage={"model": "openai/gpt-5", "input_tokens": 0, "output_tokens": 0},
    )
    assert r.meta_recommendation == "proceed"
    assert r.suggested_reformulation is None


def test_meta_frame_result_passthrough():
    """When the model fails or flag is false, callers can build a passthrough result."""
    r = MetaFrameResult(
        decision_class="unknown",
        urgency="unknown",
        question_quality="unknown",
        suggested_reformulation=None,
        meta_recommendation="proceed",
        reasoning="meta-frame skipped",
        gpt5_usage=None,
    )
    assert r.gpt5_usage is None
    assert r.meta_recommendation == "proceed"


def test_enable_meta_frame_default_true():
    # Just check the flag exists and is bool; default=True at import time.
    assert isinstance(ENABLE_META_FRAME, bool)


import json
from unittest.mock import AsyncMock, patch

from henge.providers.base import CompletionResponse
from henge.meta_frame import evaluate_question_quality


def _resp(text: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        input_tokens=200,
        output_tokens=400,
        model="openai/gpt-5",
        raw_model="gpt-5",
        finish_reason="stop",
    )


_PROCEED_JSON = json.dumps({
    "decision_class": "two-way-with-cost",
    "urgency": "weeks",
    "question_quality": "well-formed",
    "suggested_reformulation": None,
    "meta_recommendation": "proceed",
    "reasoning": "The question is binary, has clear stakes, and the user provided context that grounds it.",
})

_REFORMULATE_JSON = json.dumps({
    "decision_class": "reversible",
    "urgency": "fake-urgency",
    "question_quality": "exploration-disguised-as-decision",
    "suggested_reformulation": "What would you learn in 30 days that would commit you to one path or the other?",
    "meta_recommendation": "reformulate",
    "reasoning": "The question conflates exploration with decision. Without naming what would change your mind, no answer can be load-bearing.",
})


@pytest.mark.asyncio
async def test_evaluate_proceed_path():
    async def fake_complete(model_id, req):
        assert model_id == "openai/gpt-5"
        return _resp(_PROCEED_JSON)

    with patch("henge.meta_frame.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await evaluate_question_quality("Q?", "ctx")

    assert r.meta_recommendation == "proceed"
    assert r.question_quality == "well-formed"
    assert r.suggested_reformulation is None
    assert r.gpt5_usage is not None
    assert r.gpt5_usage["model"] == "openai/gpt-5"


@pytest.mark.asyncio
async def test_evaluate_reformulate_path():
    async def fake_complete(model_id, req):
        return _resp(_REFORMULATE_JSON)

    with patch("henge.meta_frame.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await evaluate_question_quality("Q?", "ctx")

    assert r.meta_recommendation == "reformulate"
    assert r.suggested_reformulation is not None
    assert "30 days" in r.suggested_reformulation


@pytest.mark.asyncio
async def test_evaluate_handles_garbage_json():
    async def fake_complete(model_id, req):
        return _resp("not json at all, just prose")

    with patch("henge.meta_frame.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await evaluate_question_quality("Q?", "ctx")

    # Degraded: returns proceed with unknown classifications
    assert r.meta_recommendation == "proceed"
    assert r.question_quality == "unknown"
    assert r.gpt5_usage is not None  # call happened, parse failed


@pytest.mark.asyncio
async def test_evaluate_skips_when_flag_false(monkeypatch):
    monkeypatch.setattr("henge.meta_frame.ENABLE_META_FRAME", False)
    called = []

    async def fake_complete(model_id, req):
        called.append(model_id)
        return _resp(_PROCEED_JSON)

    with patch("henge.meta_frame.complete", new=AsyncMock(side_effect=fake_complete)):
        r = await evaluate_question_quality("Q?", "ctx")

    assert called == []
    assert r.meta_recommendation == "proceed"
    assert r.gpt5_usage is None
