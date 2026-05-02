"""Phase 3 scoping pipeline: ScopingResult, CanonicalContext, feature flags."""
import pytest

from henge.scoping import (
    ScopingResult,
    CanonicalContext,
    ScopedQuestion,
)


def test_scoping_result_is_dataclass_like():
    questions = [
        ScopedQuestion(id="q_001", text="Q?", source="scoping", challenges_assumption=None),
    ]
    r = ScopingResult(
        questions=questions,
        adversarial_count=0,
        version="v0.6",
        haiku_usage={"model": "anthropic/haiku-4-5", "input_tokens": 0, "output_tokens": 0},
        gpt5_usage=None,
    )
    assert r.adversarial_count == 0
    assert r.version == "v0.6"
    assert r.questions[0].source == "scoping"


def test_canonical_context_is_dataclass_like():
    c = CanonicalContext(
        summary="The user is a 30-year-old considering X.",
        flags=[],
        opus_usage={"model": "anthropic/opus-4-7", "input_tokens": 0, "output_tokens": 0},
    )
    assert c.summary.startswith("The user")
    assert c.flags == []


def test_scoped_question_source_must_be_known():
    ScopedQuestion(id="q_002", text="Q", source="adversarial", challenges_assumption="X assumes...")
    ScopedQuestion(id="q_003", text="Q", source="scoping", challenges_assumption=None)


import pytest
from unittest.mock import AsyncMock, patch

from henge.providers.base import CompletionResponse
from henge.scoping import run_scoping


def _resp(text: str, model_id: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        input_tokens=120,
        output_tokens=60,
        model=model_id,
        raw_model=model_id.split("/", 1)[1],
        finish_reason="end_turn",
    )


_HAIKU_OK = '["Q1?","Q2?","Q3?","Q4?"]'
_GPT5_OK = (
    '[{"text":"Why are you assuming X?","challenges_assumption":"that X is fixed"},'
    ' {"text":"What if Y is wrong?","challenges_assumption":"that Y must hold"}]'
)


@pytest.mark.asyncio
async def test_run_scoping_combines_base_and_adversarial():
    async def fake_complete(model_id, req):
        if model_id == "anthropic/haiku-4-5":
            return _resp(_HAIKU_OK, model_id)
        if model_id == "openai/gpt-5":
            return _resp(_GPT5_OK, model_id)
        raise AssertionError(f"unexpected model {model_id}")

    with patch("henge.scoping.complete", new=AsyncMock(side_effect=fake_complete)):
        result = await run_scoping("¿Debo cerrar mi consultora?")

    assert len(result.questions) == 6  # 4 base + 2 adversarial
    assert result.adversarial_count == 2
    assert result.version == "v0.6"

    base = [q for q in result.questions if q.source == "scoping"]
    adv = [q for q in result.questions if q.source == "adversarial"]
    assert len(base) == 4
    assert len(adv) == 2
    assert base[0].id == "q_001"
    assert all(q.challenges_assumption is None for q in base)
    assert all(q.challenges_assumption is not None for q in adv)


@pytest.mark.asyncio
async def test_run_scoping_skips_adversarial_when_flag_false(monkeypatch):
    monkeypatch.setattr("henge.scoping.ENABLE_ADVERSARIAL", False)
    haiku_calls = []

    async def fake_complete(model_id, req):
        haiku_calls.append(model_id)
        return _resp(_HAIKU_OK, model_id)

    with patch("henge.scoping.complete", new=AsyncMock(side_effect=fake_complete)):
        result = await run_scoping("X?")

    assert result.adversarial_count == 0
    assert haiku_calls == ["anthropic/haiku-4-5"]  # gpt-5 NOT called
    assert all(q.source == "scoping" for q in result.questions)


@pytest.mark.asyncio
async def test_run_scoping_handles_haiku_json_error():
    async def fake_complete(model_id, req):
        if model_id == "anthropic/haiku-4-5":
            return _resp("not json at all", model_id)
        return _resp(_GPT5_OK, model_id)

    with patch("henge.scoping.complete", new=AsyncMock(side_effect=fake_complete)):
        result = await run_scoping("X?")

    # Degraded: zero base questions; adversarial still runs (gpt-5 sees the original question).
    assert result.adversarial_count == 2
    assert len([q for q in result.questions if q.source == "scoping"]) == 0


@pytest.mark.asyncio
async def test_finalize_context_returns_summary():
    from henge.scoping import finalize_context

    async def fake_complete(model_id, req):
        assert model_id == "anthropic/opus-4-7"
        return _resp(
            "User is X with constraints Y and Z. Inconsistency: claim A vs claim B.",
            model_id,
        )

    with patch("henge.scoping.complete", new=AsyncMock(side_effect=fake_complete)):
        ctx = await finalize_context("Original Q", "User answered: A=1, B=2.")

    assert "User is X" in ctx.summary
    assert ctx.opus_usage is not None
    assert ctx.opus_usage["model"] == "anthropic/opus-4-7"


@pytest.mark.asyncio
async def test_finalize_context_skips_when_flag_false(monkeypatch):
    from henge.scoping import finalize_context

    monkeypatch.setattr("henge.scoping.ENABLE_CANONICAL_CONTEXT", False)
    called = []

    async def fake_complete(model_id, req):
        called.append(model_id)
        return _resp("UNUSED", model_id)

    with patch("henge.scoping.complete", new=AsyncMock(side_effect=fake_complete)):
        ctx = await finalize_context("Q", "raw user context here")

    assert called == []  # Opus not invoked
    assert ctx.summary == "raw user context here"  # passthrough
    assert ctx.opus_usage is None
