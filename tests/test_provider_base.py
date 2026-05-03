"""ABC contract tests. ProviderBase no instanciable; subclase debe implementar 3 métodos."""
import pytest
from henge.providers.base import (
    ProviderBase,
    CompletionRequest,
    CompletionResponse,
)


def test_provider_base_is_abstract():
    with pytest.raises(TypeError):
        ProviderBase()


def test_completion_request_dataclass_defaults():
    req = CompletionRequest(system="s", user="u", max_tokens=100)
    assert req.temperature == 0.0


def test_completion_response_fields():
    r = CompletionResponse(
        text="hi",
        input_tokens=10,
        output_tokens=2,
        model="anthropic/opus-4-7",
        raw_model="claude-opus-4-7",
        finish_reason="end_turn",
    )
    assert r.model == "anthropic/opus-4-7"
    assert r.raw_model == "claude-opus-4-7"


def test_subclass_must_implement_complete_and_supports_and_cost():
    class Partial(ProviderBase):
        async def complete(self, model_id, req):
            return None

    with pytest.raises(TypeError):
        Partial()
