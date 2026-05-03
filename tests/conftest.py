"""Shared mocks for Henge test suite.

Why mock heavily: tests must run in <5s and never hit real APIs (no tokens spent).
"""
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_anthropic_client():
    """AsyncAnthropic-shaped mock that returns deterministic responses per frame."""
    client = MagicMock()

    async def mock_create(**kwargs):
        system = str(kwargs.get("system", ""))
        messages = kwargs.get("messages", [])
        user_content = str(messages[0].get("content", "")) if messages else ""

        if "décimo hombre" in system.lower() or "tenth-man" in system.lower():
            response_text = (
                "Disenso steel-man: el consenso de los 9 asume X, pero si Y entonces "
                "todos están equivocados — aquí evidencia precedente Z."
            )
        elif "empírico" in system.lower():
            response_text = "Análisis empírico: base rate ~30%, fuente: estudio 2023."
        elif "histórico" in system.lower():
            response_text = "Precedente: en 2019, intento similar fracasó por A."
        elif "primer principios" in system.lower() or "first" in system.lower():
            response_text = "Primer principio: la economía dicta que B se cumple."
        elif "analógico" in system.lower():
            response_text = "Analogía: como en biología, evolución selecciona C."
        elif "sistémico" in system.lower():
            response_text = "Sistémico: feedback loop reforzante produce D."
        elif "ético" in system.lower():
            response_text = "Tensión ética: derechos vs consecuencias en E."
        elif "contrarian" in system.lower():
            response_text = "Sí pero considera que F no es necesariamente cierto."
        elif "optimista" in system.lower():
            response_text = "Caso 10×: si G ocurre, se desbloquea H."
        elif "pre-mortem" in system.lower():
            response_text = "Modo de falla: equipo no pudo cerrar I porque J."
        else:
            response_text = f"Respuesta genérica (system={system[:30]})."

        result = MagicMock()
        text_part = MagicMock()
        text_part.text = response_text
        result.content = [text_part]
        # Real Anthropic responses populate ``usage`` with token counts.
        # Tests need explicit ints so cost accounting (pricing.total_cost)
        # gets real numbers instead of MagicMock auto-attrs.
        result.usage = MagicMock(input_tokens=120, output_tokens=80)
        return result

    client.messages.create = AsyncMock(side_effect=mock_create)
    return client


@pytest.fixture
def synthetic_embeddings_10():
    """10 synthetic embeddings: 9 cluster tightly, 1 outlier far away.

    Used to verify that centroid_of_9 excludes the 10th — if the centroid
    accidentally includes #10, distances get diluted and the test catches it.
    """
    rng = np.random.default_rng(42)
    base = rng.normal(0, 0.1, size=(9, 1024))
    base[:, 0] += 1.0  # bias toward positive direction
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    outlier = rng.normal(0, 0.1, size=(1, 1024))
    outlier[:, 0] -= 1.0  # bias toward opposite direction
    outlier = outlier / np.linalg.norm(outlier, axis=1, keepdims=True)
    return np.vstack([base, outlier]).tolist()


from unittest.mock import patch as _patch

from henge.providers.base import CompletionResponse as _CompletionResponse


def _mock_complete_factory():
    """Return an async function that simulates ``providers.complete`` per frame."""

    async def _mock_complete(model_id, req):
        system = (req.system or "").lower()

        # Informed tenth-man (gpt-5): needs JSON-shaped response for json.loads()
        if "audit a blind dissent" in system or "blind tenth-man dissent" in system:
            text = '{"text":"refined dissent in mock","what_holds":[],"what_revised":[],"what_discarded":[]}'
            return _CompletionResponse(
                text=text,
                input_tokens=120,
                output_tokens=80,
                model=model_id,
                raw_model=model_id.split("/", 1)[1] if "/" in model_id else model_id,
                finish_reason="end_turn",
            )

        if "empírico" in system or "empirical" in system:
            text = "Análisis empírico: base rate ~30%, fuente: estudio 2023."
        elif "histórico" in system or "historical" in system:
            text = "Precedente: en 2019, intento similar fracasó por A."
        elif "primer principios" in system or "first principles" in system or "first-principles" in system:
            text = "Primer principio: la economía dicta que B se cumple."
        elif "analógico" in system or "analogical" in system:
            text = "Analogía: como en biología, evolución selecciona C."
        elif "sistémico" in system or "systemic" in system:
            text = "Sistémico: feedback loop reforzante produce D."
        elif "ético" in system or "ethical" in system:
            text = "Tensión ética: derechos vs consecuencias en E."
        elif "contrarian" in system:
            text = "Sí pero considera que F no es necesariamente cierto."
        elif "optimist" in system:
            text = "Caso 10×: si G ocurre, se desbloquea H."
        elif "pre-mortem" in system or "premortem" in system:
            text = "Modo de falla: equipo no pudo cerrar I porque J."
        else:
            text = f"Respuesta genérica (model={model_id})."

        return _CompletionResponse(
            text=text,
            input_tokens=120,
            output_tokens=80,
            model=model_id,
            raw_model=model_id.split("/", 1)[1] if "/" in model_id else model_id,
            finish_reason="end_turn",
        )

    return _mock_complete


@pytest.fixture
def mock_providers():
    """Patch ``henge.agents.complete`` and ``henge.tenth_man.complete`` so all
    provider calls (9 frames + blind + informed) return deterministic text.

    Both modules bind ``complete`` at import time, so we must patch both
    references to intercept every call. Yields the AsyncMock instance so
    individual tests can install side_effects (e.g. raise on the 3rd call to
    simulate a frame failure).
    """
    mock = AsyncMock(side_effect=_mock_complete_factory())
    with _patch("henge.agents.complete", new=mock), \
         _patch("henge.tenth_man.complete", new=mock):
        yield mock
