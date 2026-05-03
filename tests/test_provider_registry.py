"""Registry: anthropic/* → AnthropicProvider, openai/* → OpenAIProvider, otros → ValueError."""
import pytest

from henge.providers import (
    CompletionRequest,
    complete,
    cost_usd,
    get_provider_for,
)
from henge.providers.anthropic_provider import AnthropicProvider
from henge.providers.openai_provider import OpenAIProvider


def test_resolves_anthropic():
    assert isinstance(get_provider_for("anthropic/sonnet-4-6"), AnthropicProvider)


def test_resolves_openai():
    assert isinstance(get_provider_for("openai/gpt-5"), OpenAIProvider)


def test_unknown_raises():
    with pytest.raises(ValueError):
        get_provider_for("google/gemini-3")


def test_singleton_reuse():
    a = get_provider_for("anthropic/opus-4-7")
    b = get_provider_for("anthropic/sonnet-4-6")
    assert a is b  # same anthropic singleton


def test_cost_usd_via_registry():
    assert cost_usd("anthropic/haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert cost_usd("openai/gpt-5", 1_000_000, 0) == pytest.approx(5.00)
    assert cost_usd("unknown", 999, 999) == 0.0
