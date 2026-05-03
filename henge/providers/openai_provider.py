"""OpenAIProvider — wraps openai.AsyncOpenAI with canonical id mapping.

If ``gpt-5`` is not yet available at implementation time, update ``_RAW_MODEL``
to point ``openai/gpt-5`` to the frontier model actually shipped (e.g.
``gpt-5-mini`` or current frontier). The canonical id is the contract — do
NOT change it elsewhere.

OpenAI normalisation notes:
- ``messages`` is system+user (no system kwarg like Anthropic).
- ``max_completion_tokens`` replaces ``max_tokens`` for newer models.
- Token usage exposed as ``prompt_tokens``/``completion_tokens``.
- ``finish_reason`` lives on each choice.
- ``temperature`` is omitted when ``req.temperature == 0.0`` (the
  ``CompletionRequest`` default). Reasoning models (gpt-5) reject an explicit
  0.0; chat models silently fall through to their API default (1). Callers
  needing strict 0.0 with chat models would require an API-rejected fallback
  similar to ``anthropic_provider.py``'s try/except — out of scope for v0.6.
"""
from __future__ import annotations

from typing import Any

from henge.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ProviderBase,
)
from henge.providers.pricing import cost_for

_RAW_MODEL = {
    "openai/gpt-5": "gpt-5",
}


class OpenAIProvider(ProviderBase):
    def __init__(self, client: Any | None = None):
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
        self._client = client

    def supports(self, model_id: str) -> bool:
        return model_id in _RAW_MODEL

    async def complete(
        self, model_id: str, req: CompletionRequest
    ) -> CompletionResponse:
        if not self.supports(model_id):
            raise ValueError(f"OpenAIProvider does not support {model_id}")
        raw = _RAW_MODEL[model_id]

        kwargs: dict = dict(
            model=raw,
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            max_completion_tokens=req.max_tokens,
        )
        # Only forward temperature when explicitly non-default; gpt-5 and
        # similar frontier models reject temperature=0.0 (they support only
        # the API default of 1).
        if req.temperature != 0.0:
            kwargs["temperature"] = req.temperature

        # gpt-5 reasoning_effort caps internal chain-of-thought tokens.
        # "low" is enough for structured-output frame tasks; the default
        # ("medium") burns 2000+ reasoning tokens before any visible output.
        if req.reasoning_effort is not None:
            kwargs["reasoning_effort"] = req.reasoning_effort

        completion = await self._client.chat.completions.create(**kwargs)

        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        return CompletionResponse(
            text=choice.message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=model_id,
            raw_model=raw,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        )

    def cost_usd(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        return cost_for(model_id, input_tokens, output_tokens)
