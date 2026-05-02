"""Resolución model_id → provider singleton. Stub; full impl en Task 1.5."""
from henge.providers.base import CompletionRequest, CompletionResponse, ProviderBase


def get_provider_for(model_id: str) -> ProviderBase:
    raise NotImplementedError("filled in Task 1.5")


async def complete(model_id: str, req: CompletionRequest) -> CompletionResponse:
    raise NotImplementedError("filled in Task 1.5")


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    raise NotImplementedError("filled in Task 1.5")
