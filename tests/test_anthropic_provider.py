"""AnthropicProvider mappea canonical_id → raw_model y normaliza la respuesta."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from henge.providers.anthropic_provider import AnthropicProvider
from henge.providers.base import CompletionRequest


@pytest.fixture
def fake_client():
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text="hello")]
    msg.usage = MagicMock(input_tokens=42, output_tokens=7)
    msg.stop_reason = "end_turn"
    client.messages.create = AsyncMock(return_value=msg)
    return client


def test_supports_anthropic_ids():
    p = AnthropicProvider(client=MagicMock())
    assert p.supports("anthropic/sonnet-4-6")
    assert p.supports("anthropic/opus-4-7")
    assert not p.supports("openai/gpt-5")


def test_supports_rejects_unknown_anthropic_id():
    p = AnthropicProvider(client=MagicMock())
    assert not p.supports("anthropic/llama-9999")


@pytest.mark.asyncio
async def test_complete_maps_canonical_to_raw_and_extracts(fake_client):
    p = AnthropicProvider(client=fake_client)
    req = CompletionRequest(system="s", user="u", max_tokens=100)
    resp = await p.complete("anthropic/sonnet-4-6", req)

    fake_client.messages.create.assert_awaited_once()
    call_kwargs = fake_client.messages.create.await_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["temperature"] == 0.0

    assert resp.text == "hello"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 7
    assert resp.model == "anthropic/sonnet-4-6"
    assert resp.raw_model == "claude-sonnet-4-6"
    assert resp.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_complete_omits_temperature_for_opus(fake_client):
    """Opus 4.7 (extended-thinking tier) rejects temperature param."""
    p = AnthropicProvider(client=fake_client)
    req = CompletionRequest(system="s", user="u", max_tokens=100, temperature=0.7)
    await p.complete("anthropic/opus-4-7", req)
    call_kwargs = fake_client.messages.create.await_args.kwargs
    assert "temperature" not in call_kwargs


def test_cost_usd_uses_pricing_table():
    p = AnthropicProvider(client=MagicMock())
    # Haiku @ 1M in / 0 out = $1.00
    assert p.cost_usd("anthropic/haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
