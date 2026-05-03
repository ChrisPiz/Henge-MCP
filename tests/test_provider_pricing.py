"""Pricing es dato versionado. Cualquier cambio bumpea PRICING_VERSION."""
import pytest

from henge.providers.pricing import (
    PRICING,
    PRICING_VERSION,
    cost_for,
)


def test_pricing_version_is_iso_month():
    # YYYY-MM, no day. Bump on every price change.
    parts = PRICING_VERSION.split("-")
    assert len(parts) == 2
    assert len(parts[0]) == 4 and parts[0].isdigit()
    assert len(parts[1]) == 2 and parts[1].isdigit()


def test_required_models_priced():
    required = [
        "anthropic/haiku-4-5",
        "anthropic/sonnet-4-6",
        "anthropic/opus-4-7",
        "openai/gpt-5",
    ]
    for m in required:
        assert m in PRICING, f"missing pricing for {m}"
        assert PRICING[m]["in"] > 0
        assert PRICING[m]["out"] >= 0


def test_cost_for_known_model():
    # Sonnet @ 1k in / 1k out = (1000*3 + 1000*15) / 1e6 = 0.018
    cost = cost_for("anthropic/sonnet-4-6", 1000, 1000)
    assert cost == pytest.approx(0.018, abs=1e-9)


def test_cost_for_unknown_returns_zero():
    assert cost_for("unknown/model", 1000, 1000) == 0.0


def test_cost_for_negative_safe():
    assert cost_for("anthropic/opus-4-7", -1, -1) == 0.0
