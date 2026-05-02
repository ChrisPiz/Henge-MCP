"""AnthropicProvider — wraps anthropic.AsyncAnthropic with canonical id mapping.

The mapping ``canonical_id → raw_model`` is the only place SDK strings appear
outside this file. Models in ``_NO_TEMPERATURE`` are reasoning-tier (Opus 4.7
with extended thinking) — they reject the ``temperature`` kwarg outright, so
we omit it.
"""
from __future__ import annotations

from typing import Any

from henge.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ProviderBase,
)
from henge.providers.pricing import cost_for

# canonical id -> SDK model string
_RAW_MODEL = {
    "anthropic/haiku-4-5":  "claude-haiku-4-5-20251001",
    "anthropic/sonnet-4-6": "claude-sonnet-4-6",
    "anthropic/opus-4-7":   "claude-opus-4-7",
}

_NO_TEMPERATURE = {"anthropic/opus-4-7"}


class AnthropicProvider(ProviderBase):
    def __init__(self, client: Any | None = None):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
        self._client = client

    def supports(self, model_id: str) -> bool:
        return model_id in _RAW_MODEL

    async def complete(
        self, model_id: str, req: CompletionRequest
    ) -> CompletionResponse:
        if not self.supports(model_id):
            raise ValueError(f"AnthropicProvider does not support {model_id}")
        raw = _RAW_MODEL[model_id]

        kwargs = {
            "model": raw,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
        }
        if model_id not in _NO_TEMPERATURE:
            kwargs["temperature"] = req.temperature

        try:
            msg = await self._client.messages.create(**kwargs)
        except Exception as exc:
            err = str(exc).lower()
            temperature_rejected = any(
                kw in err
                for kw in ("temperature", "extended thinking", "thinking is enabled")
            )
            if not temperature_rejected:
                raise
            kwargs.pop("temperature", None)
            msg = await self._client.messages.create(**kwargs)

        usage = getattr(msg, "usage", None)
        return CompletionResponse(
            text=msg.content[0].text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            model=model_id,
            raw_model=raw,
            finish_reason=str(getattr(msg, "stop_reason", "") or ""),
        )

    def cost_usd(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        return cost_for(model_id, input_tokens, output_tokens)
