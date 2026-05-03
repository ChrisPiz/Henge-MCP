"""Phase 8 cost_breakdown — canonical-id-aware aggregation across all v0.6 phases."""
import pytest

from henge.providers.pricing import build_cost_breakdown, PRICING_VERSION


def _u(model_id: str, inp: int, out: int) -> dict:
    return {"model": model_id, "input_tokens": inp, "output_tokens": out}


def test_anthropic_only_run():
    breakdown = build_cost_breakdown(
        advisor_usages=[_u("anthropic/sonnet-4-6", 1_000_000, 1_000_000)],
        blind_usage=_u("anthropic/opus-4-7", 1_000_000, 1_000_000),
        informed_usage=None,
        meta_usage=None,
        canonical_usage=None,
        scoping_haiku_usage=None,
        scoping_adversarial_usage=None,
        consensus_usage=None,
        claims_extract_usage=None,
        claims_verify_usage=None,
        embedding_model="openai/text-embedding-3-large",
        embedding_input_tokens=0,
    )
    # Sonnet 1M+1M = 3+15 = 18, Opus 1M+1M = 15+75 = 90
    assert breakdown["anthropic_usd"] == pytest.approx(108.0)
    assert breakdown["openai_usd"] == pytest.approx(0.0)
    assert breakdown["embedding_usd"] == pytest.approx(0.0)
    assert breakdown["total_usd"] == pytest.approx(108.0)
    assert breakdown["pricing_version"] == PRICING_VERSION


def test_mixed_anthropic_openai_run():
    breakdown = build_cost_breakdown(
        advisor_usages=[
            _u("anthropic/sonnet-4-6", 1000, 500),    # 0.003 + 0.0075 = 0.0105
            _u("openai/gpt-5", 1000, 500),            # 0.005 + 0.010 = 0.015
        ],
        blind_usage=_u("anthropic/opus-4-7", 2000, 1500),  # 0.030 + 0.1125 = 0.1425
        informed_usage=_u("openai/gpt-5", 1000, 500),       # 0.005 + 0.010 = 0.015
        meta_usage=_u("openai/gpt-5", 500, 500),            # 0.0025 + 0.010 = 0.0125
        canonical_usage=_u("anthropic/opus-4-7", 500, 500), # 0.0075 + 0.0375 = 0.045
        scoping_haiku_usage=_u("anthropic/haiku-4-5", 500, 200),  # 0.0005 + 0.001 = 0.0015
        scoping_adversarial_usage=_u("openai/gpt-5", 500, 1000),  # 0.0025 + 0.020 = 0.0225
        consensus_usage={"model": "claude-haiku-4-5-20251001", "input_tokens": 800, "output_tokens": 400},  # legacy raw string
        claims_extract_usage=_u("anthropic/sonnet-4-6", 500, 300),  # 0.0015 + 0.0045 = 0.006
        claims_verify_usage=_u("openai/gpt-5", 1000, 500),          # 0.005 + 0.010 = 0.015
        embedding_model="openai/text-embedding-3-large",
        embedding_input_tokens=10_000,  # 10k * 0.13 / 1M = 0.0013
    )
    # Anthropic sum:
    #   Sonnet frame:    0.0105
    #   Opus blind:      0.1425
    #   Opus canonical:  0.045
    #   Haiku scoping:   0.0015
    #   Haiku consensus: 0.0008 + 0.002 = 0.0028 (800 * 1 + 400 * 5 = 2800 / 1e6 = 0.0028)
    #   Sonnet claims:   0.006
    # Total anthropic: ~0.2078
    # OpenAI sum:
    #   gpt-5 frame:     0.015
    #   gpt-5 informed:  0.015
    #   gpt-5 meta:      0.0125
    #   gpt-5 scoping:   0.0225
    #   gpt-5 verify:    0.015
    # Total openai: ~0.08
    # Embedding: 10000 * 0.13 / 1e6 = 0.0013
    assert breakdown["anthropic_usd"] > 0
    assert breakdown["openai_usd"] > 0
    assert breakdown["embedding_usd"] == pytest.approx(0.0013)
    assert breakdown["anthropic_usd"] == pytest.approx(0.2078, abs=1e-3)
    assert breakdown["openai_usd"] == pytest.approx(0.08, abs=1e-3)
    assert breakdown["total_usd"] == pytest.approx(
        breakdown["anthropic_usd"] + breakdown["openai_usd"] + breakdown["embedding_usd"],
        abs=1e-9,
    )


def test_handles_none_usages_gracefully():
    breakdown = build_cost_breakdown(
        advisor_usages=[None, None],
        blind_usage=None,
        informed_usage=None,
        meta_usage=None,
        canonical_usage=None,
        scoping_haiku_usage=None,
        scoping_adversarial_usage=None,
        consensus_usage=None,
        claims_extract_usage=None,
        claims_verify_usage=None,
        embedding_model="openai/text-embedding-3-large",
        embedding_input_tokens=0,
    )
    assert breakdown["anthropic_usd"] == 0.0
    assert breakdown["openai_usd"] == 0.0
    assert breakdown["total_usd"] == 0.0


def test_consensus_legacy_raw_model_normalized():
    """consensus_usage may carry the v0.5 raw model string. The builder must normalize it."""
    breakdown = build_cost_breakdown(
        advisor_usages=[],
        blind_usage=None,
        informed_usage=None,
        meta_usage=None,
        canonical_usage=None,
        scoping_haiku_usage=None,
        scoping_adversarial_usage=None,
        consensus_usage={"model": "claude-haiku-4-5-20251001", "input_tokens": 1_000_000, "output_tokens": 1_000_000},
        claims_extract_usage=None,
        claims_verify_usage=None,
        embedding_model="openai/text-embedding-3-large",
        embedding_input_tokens=0,
    )
    # haiku 1M + 1M = 1.0 + 5.0 = 6.0
    assert breakdown["anthropic_usd"] == pytest.approx(6.0)
    assert breakdown["openai_usd"] == 0.0
