"""Singleton registry mapping canonical model id prefix → provider instance.

Lazy-init: providers are not constructed until first use, so importing
``henge.providers`` does not require API keys or instantiate SDK clients.
"""
from __future__ import annotations

from threading import Lock

from henge.providers.anthropic_provider import AnthropicProvider
from henge.providers.base import CompletionRequest, CompletionResponse, ProviderBase
from henge.providers.openai_provider import OpenAIProvider

_anthropic: AnthropicProvider | None = None
_openai: OpenAIProvider | None = None
_lock = Lock()


def _anthropic_singleton() -> AnthropicProvider:
    global _anthropic
    with _lock:
        if _anthropic is None:
            _anthropic = AnthropicProvider()
    return _anthropic


def _openai_singleton() -> OpenAIProvider:
    global _openai
    with _lock:
        if _openai is None:
            _openai = OpenAIProvider()
    return _openai


def get_provider_for(model_id: str) -> ProviderBase:
    if model_id.startswith("anthropic/"):
        return _anthropic_singleton()
    if model_id.startswith("openai/"):
        return _openai_singleton()
    raise ValueError(f"Unknown model id: {model_id}")


async def complete(model_id: str, req: CompletionRequest) -> CompletionResponse:
    return await get_provider_for(model_id).complete(model_id, req)


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    try:
        return get_provider_for(model_id).cost_usd(model_id, input_tokens, output_tokens)
    except ValueError:
        return 0.0
