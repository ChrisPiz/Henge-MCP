"""FRAME_MODEL_MAP es contrato. Cada frame asignado a un canonical_id existente."""
import pytest

from henge.config.frame_assignment import FRAME_MODEL_MAP, model_for


_EXPECTED = {
    "empirical":         "openai/gpt-5",
    "historical":        "anthropic/opus-4-7",
    "first-principles":  "openai/gpt-5",
    "analogical":        "anthropic/sonnet-4-6",
    "systemic":          "openai/gpt-5",
    "ethical":           "anthropic/sonnet-4-6",
    "soft-contrarian":   "openai/gpt-5",
    "radical-optimist":  "openai/gpt-5",
    "pre-mortem":        "openai/gpt-5",
}


def test_all_nine_frames_present():
    assert set(FRAME_MODEL_MAP.keys()) == set(_EXPECTED.keys())
    assert len(FRAME_MODEL_MAP) == 9


def test_assignment_matches_v06_spec():
    assert FRAME_MODEL_MAP == _EXPECTED


def test_distribution_is_2_sonnet_1_opus_6_gpt5():
    counts = {}
    for model in FRAME_MODEL_MAP.values():
        counts[model] = counts.get(model, 0) + 1
    assert counts["anthropic/sonnet-4-6"] == 2
    assert counts["anthropic/opus-4-7"] == 1
    assert counts["openai/gpt-5"] == 6


def test_model_for_returns_canonical_id():
    assert model_for("empirical") == "openai/gpt-5"
    assert model_for("historical") == "anthropic/opus-4-7"


def test_model_for_unknown_frame_raises():
    with pytest.raises(KeyError):
        model_for("not-a-frame")


def test_models_are_resolvable_by_registry():
    """Cada model_id en FRAME_MODEL_MAP debe ser conocido por el registry."""
    from henge.providers import get_provider_for
    for frame, model_id in FRAME_MODEL_MAP.items():
        provider = get_provider_for(model_id)
        assert provider.supports(model_id), f"{frame} → {model_id} not supported"
