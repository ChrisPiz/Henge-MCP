"""OpenAIProvider mappea canonical_id → raw_model y normaliza al mismo CompletionResponse."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from henge.providers.openai_provider import OpenAIProvider
from henge.providers.base import CompletionRequest


@pytest.fixture
def fake_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message = MagicMock(content="hi from gpt")
    choice.finish_reason = "stop"
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(prompt_tokens=11, completion_tokens=4)
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


def test_supports():
    p = OpenAIProvider(client=MagicMock())
    assert p.supports("openai/gpt-5")
    assert not p.supports("anthropic/opus-4-7")


@pytest.mark.asyncio
async def test_complete_maps_and_normalizes(fake_client):
    p = OpenAIProvider(client=fake_client)
    req = CompletionRequest(system="sys", user="usr", max_tokens=200, temperature=0.0)
    resp = await p.complete("openai/gpt-5", req)

    call = fake_client.chat.completions.create.await_args.kwargs
    assert call["model"] == "gpt-5"
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["messages"][1] == {"role": "user", "content": "usr"}
    assert call["max_completion_tokens"] == 200

    assert resp.text == "hi from gpt"
    assert resp.input_tokens == 11
    assert resp.output_tokens == 4
    assert resp.model == "openai/gpt-5"
    assert resp.raw_model == "gpt-5"
    assert resp.finish_reason == "stop"


def test_cost_usd():
    p = OpenAIProvider(client=MagicMock())
    # gpt-5 @ 1M in / 1M out = 5 + 20 = 25
    assert p.cost_usd("openai/gpt-5", 1_000_000, 1_000_000) == pytest.approx(25.0)
