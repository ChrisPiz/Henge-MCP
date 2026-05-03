"""Real-call smoke test for both providers. Skipped automatically without API keys.

Anthropic call uses max_tokens=10 (~USD 0.001). OpenAI gpt-5 uses
max_tokens=500 because it is a reasoning model that burns ~100+ internal
tokens before output (~USD 0.01–0.05 per run). Verifies that
the canonical interface works end-to-end against live SDKs.
"""
import os

import pytest

from henge.providers import CompletionRequest, complete

def _is_real_key(name: str) -> bool:
    val = os.getenv(name, "")
    if not val:
        return False
    return "test" not in val.lower() and "stub" not in val.lower()


_HAS_ANTHROPIC = _is_real_key("ANTHROPIC_API_KEY")
_HAS_OPENAI = _is_real_key("OPENAI_API_KEY")


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
async def test_smoke_anthropic_haiku():
    req = CompletionRequest(
        system="Reply with just the digit 4.",
        user="What is 2+2?",
        max_tokens=10,
    )
    resp = await complete("anthropic/haiku-4-5", req)
    assert resp.text.strip() != ""
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.raw_model.startswith("claude-haiku-4-5")


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_OPENAI, reason="OPENAI_API_KEY not set")
async def test_smoke_openai_gpt5():
    # gpt-5 is a reasoning model; it consumes ~100+ tokens internally before
    # producing any output, so max_tokens=500 is the practical minimum.
    req = CompletionRequest(
        system="Reply with just the digit 4.",
        user="What is 2+2?",
        max_tokens=500,
    )
    resp = await complete("openai/gpt-5", req)
    assert resp.text.strip() != ""
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    # Provider stores the bare alias from _RAW_MODEL ("gpt-5"); the SDK's
    # completion.model is dated but is not what we surface. startswith is
    # defensive against any future change to the canonical→raw mapping.
    assert resp.raw_model.startswith("gpt-5")
