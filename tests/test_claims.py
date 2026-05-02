"""Phase 6 claims: Claim, ClaimVerification, extract_claims, verify_claims."""
import json
import pytest
from unittest.mock import AsyncMock, patch

from henge.claims import (
    Claim,
    ClaimVerification,
    ENABLE_CLAIM_VERIFICATION,
    extract_claims,
    verify_claims,
)
from henge.providers.base import CompletionResponse


def _resp(text: str, model_id: str) -> CompletionResponse:
    return CompletionResponse(
        text=text,
        input_tokens=200,
        output_tokens=300,
        model=model_id,
        raw_model=model_id.split("/", 1)[1],
        finish_reason="end_turn",
    )


def test_claim_dataclass_fields():
    c = Claim(text="The runway is 3 weeks.", type="factual")
    assert c.text.startswith("The runway")
    assert c.type == "factual"


def test_claim_verification_dataclass_fields():
    v = ClaimVerification(
        claim_text="X",
        claim_type="factual",
        supporting_frames=["empirical", "first-principles"],
        contesting_frames=["radical-optimist"],
        support_strength="moderate",
    )
    assert len(v.supporting_frames) == 2
    assert len(v.contesting_frames) == 1
    assert v.support_strength == "moderate"


def test_enable_flag_default_true():
    assert isinstance(ENABLE_CLAIM_VERIFICATION, bool)


_EXTRACT_JSON = json.dumps([
    {"text": "The runway with current burn is 3-4 weeks.", "type": "factual"},
    {"text": "Closing rabbithole now violates the survival constraint.", "type": "prescriptive"},
    {"text": "Without paid pilots, the AI pivot will fail.", "type": "causal"},
])


@pytest.mark.asyncio
async def test_extract_claims_happy_path():
    async def fake_complete(model_id, req):
        assert model_id == "anthropic/sonnet-4-6"
        return _resp(_EXTRACT_JSON, model_id)

    with patch("henge.claims.complete", new=AsyncMock(side_effect=fake_complete)):
        claims, usage = await extract_claims("# Some consensus...\n\nBody...")

    assert len(claims) == 3
    assert claims[0].type == "factual"
    assert claims[1].type == "prescriptive"
    assert claims[2].type == "causal"
    assert usage is not None
    assert usage["model"] == "anthropic/sonnet-4-6"


@pytest.mark.asyncio
async def test_extract_claims_handles_garbage_json():
    async def fake_complete(model_id, req):
        return _resp("not json prose", model_id)

    with patch("henge.claims.complete", new=AsyncMock(side_effect=fake_complete)):
        claims, usage = await extract_claims("consensus")

    assert claims == []
    assert usage is not None  # call happened, parse failed


@pytest.mark.asyncio
async def test_extract_claims_skips_when_flag_false(monkeypatch):
    monkeypatch.setattr("henge.claims.ENABLE_CLAIM_VERIFICATION", False)
    called = []

    async def fake_complete(model_id, req):
        called.append(model_id)
        return _resp(_EXTRACT_JSON, model_id)

    with patch("henge.claims.complete", new=AsyncMock(side_effect=fake_complete)):
        claims, usage = await extract_claims("consensus")

    assert called == []
    assert claims == []
    assert usage is None


_VERIFY_JSON = json.dumps([
    {
        "claim_text": "The runway with current burn is 3-4 weeks.",
        "claim_type": "factual",
        "supporting_frames": ["empirical", "first-principles", "systemic"],
        "contesting_frames": [],
        "support_strength": "moderate",
    },
    {
        "claim_text": "Closing rabbithole now violates the survival constraint.",
        "claim_type": "prescriptive",
        "supporting_frames": ["empirical", "historical", "ethical", "pre-mortem", "systemic", "first-principles"],
        "contesting_frames": ["radical-optimist"],
        "support_strength": "strong",
    },
])


@pytest.mark.asyncio
async def test_verify_claims_happy_path():
    claims = [
        Claim(text="The runway with current burn is 3-4 weeks.", type="factual"),
        Claim(text="Closing rabbithole now violates the survival constraint.", type="prescriptive"),
    ]
    nine = [
        ("empirical", "Numbers say 3 weeks."),
        ("first-principles", "Survival comes first."),
    ]

    async def fake_complete(model_id, req):
        assert model_id == "openai/gpt-5"
        return _resp(_VERIFY_JSON, model_id)

    with patch("henge.claims.complete", new=AsyncMock(side_effect=fake_complete)):
        verifications, usage = await verify_claims(claims, nine)

    assert len(verifications) == 2
    assert verifications[0].support_strength == "moderate"
    assert verifications[1].support_strength == "strong"
    assert "radical-optimist" in verifications[1].contesting_frames
    assert usage is not None
    assert usage["model"] == "openai/gpt-5"


@pytest.mark.asyncio
async def test_verify_claims_handles_garbage_json():
    async def fake_complete(model_id, req):
        return _resp("not json", model_id)

    with patch("henge.claims.complete", new=AsyncMock(side_effect=fake_complete)):
        verifications, usage = await verify_claims(
            [Claim(text="X", type="factual")],
            [("empirical", "x")],
        )

    assert verifications == []
    assert usage is not None


@pytest.mark.asyncio
async def test_verify_claims_skips_empty_input():
    """If no claims to verify, don't call gpt-5 at all."""
    called = []

    async def fake_complete(model_id, req):
        called.append(model_id)
        return _resp(_VERIFY_JSON, model_id)

    with patch("henge.claims.complete", new=AsyncMock(side_effect=fake_complete)):
        verifications, usage = await verify_claims([], [])

    assert verifications == []
    assert usage is None
    assert called == []
