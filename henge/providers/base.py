"""ProviderBase ABC + request/response dataclasses.

The whole codebase outside ``henge.providers`` calls ``providers.complete(...)``
with a ``CompletionRequest`` and gets a ``CompletionResponse`` back. The
canonical model id (e.g. ``anthropic/opus-4-7``) is the contract; the raw SDK
model string lives inside each provider and never leaks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionRequest:
    system: str
    user: str
    max_tokens: int
    temperature: float = 0.0


@dataclass
class CompletionResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str           # canonical id, e.g. "anthropic/opus-4-7"
    raw_model: str       # provider-specific, e.g. "claude-opus-4-7"
    finish_reason: str


class ProviderBase(ABC):
    @abstractmethod
    async def complete(
        self, model_id: str, req: CompletionRequest
    ) -> CompletionResponse:
        ...

    @abstractmethod
    def supports(self, model_id: str) -> bool:
        ...

    @abstractmethod
    def cost_usd(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        ...
