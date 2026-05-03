"""Token pricing for Henge cost accounting (USD per 1M tokens).

Versioned data: every report records ``PRICING_VERSION`` so historical totals
remain interpretable after a price card change. Bump the version when any
``in``/``out`` value changes.

Sources:
- https://docs.anthropic.com/en/docs/about-claude/pricing
- https://platform.openai.com/docs/pricing
"""
from __future__ import annotations

PRICING_VERSION = "2026-05"

PRICING: dict[str, dict[str, float]] = {
    "anthropic/haiku-4-5":  {"in": 1.00,  "out":  5.00},
    "anthropic/sonnet-4-6": {"in": 3.00,  "out": 15.00},
    "anthropic/opus-4-7":   {"in": 15.00, "out": 75.00},
    "openai/gpt-5":         {"in": 5.00,  "out": 20.00},
}

EMBEDDING_PRICING: dict[str, float] = {
    "openai/text-embedding-3-small": 0.02,
    "openai/text-embedding-3-large": 0.13,
    "voyage/voyage-3-large":         0.18,
}


def cost_for(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one completion. Returns 0.0 for unknown models or
    negative inputs (degrade gracefully instead of failing the run)."""
    rates = PRICING.get(model_id)
    if not rates:
        return 0.0
    inp = max(0, int(input_tokens or 0))
    out = max(0, int(output_tokens or 0))
    return (inp * rates["in"] + out * rates["out"]) / 1_000_000


def embedding_cost(embedding_model_id: str, n_input_tokens: int) -> float:
    rate = EMBEDDING_PRICING.get(embedding_model_id, 0.0)
    return (max(0, int(n_input_tokens or 0)) * rate) / 1_000_000


# Map legacy raw model strings → canonical ids for back-compat with v0.5
# code paths (e.g. consensus_usage from synthesize_consensus still uses raw).
_RAW_TO_CANONICAL = {
    "claude-haiku-4-5-20251001": "anthropic/haiku-4-5",
    "claude-sonnet-4-6":         "anthropic/sonnet-4-6",
    "claude-opus-4-7":           "anthropic/opus-4-7",
    "gpt-5":                     "openai/gpt-5",
}


def _canonicalize(model: str) -> str:
    """Map a model string to its canonical id. Already-canonical ids pass through."""
    if not model:
        return ""
    if "/" in model:
        return model
    return _RAW_TO_CANONICAL.get(model, model)


def _cost_of_usage(usage: dict | None) -> tuple[float, str]:
    """Return (cost_usd, canonical_id) for a usage dict. (0.0, "") for None/unknown."""
    if not usage:
        return 0.0, ""
    canonical = _canonicalize(str(usage.get("model", "")))
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    return cost_for(canonical, inp, out), canonical


def _bucket(canonical: str) -> str:
    if canonical.startswith("anthropic/"):
        return "anthropic"
    if canonical.startswith("openai/"):
        return "openai"
    return "other"


def build_cost_breakdown(
    *,
    advisor_usages: list[dict | None],
    blind_usage: dict | None,
    informed_usage: dict | None,
    meta_usage: dict | None,
    canonical_usage: dict | None,
    scoping_haiku_usage: dict | None,
    scoping_adversarial_usage: dict | None,
    consensus_usage: dict | None,
    claims_extract_usage: dict | None,
    claims_verify_usage: dict | None,
    embedding_model: str,
    embedding_input_tokens: int = 0,
) -> dict:
    """v0.6 cost_breakdown — sums every named phase by canonical id, splits by
    provider (anthropic / openai / other), keeps embedding separate.

    Returns: {
      anthropic_usd, openai_usd, embedding_usd, total_usd, pricing_version,
      by_phase: {frames, blind, informed, meta, canonical, scoping_haiku,
                 scoping_adversarial, consensus, claims_extract, claims_verify,
                 embedding}
    }
    """
    totals = {"anthropic": 0.0, "openai": 0.0, "other": 0.0}
    by_phase: dict[str, float] = {}

    # 9 frames (or however many ran)
    frames_total = 0.0
    for u in advisor_usages or []:
        cost, canonical = _cost_of_usage(u)
        frames_total += cost
        totals[_bucket(canonical)] += cost
    by_phase["frames"] = round(frames_total, 6)

    named_calls = [
        ("blind",               blind_usage),
        ("informed",            informed_usage),
        ("meta",                meta_usage),
        ("canonical",           canonical_usage),
        ("scoping_haiku",       scoping_haiku_usage),
        ("scoping_adversarial", scoping_adversarial_usage),
        ("consensus",           consensus_usage),
        ("claims_extract",      claims_extract_usage),
        ("claims_verify",       claims_verify_usage),
    ]
    for name, u in named_calls:
        cost, canonical = _cost_of_usage(u)
        by_phase[name] = round(cost, 6)
        totals[_bucket(canonical)] += cost

    # Embedding tokens are counted separately, not in the anthropic/openai split
    # (treat embedding as its own line item — same as v0.5 convention).
    embedding_usd = round(embedding_cost(embedding_model, embedding_input_tokens), 6)
    by_phase["embedding"] = embedding_usd

    anthropic_usd = round(totals["anthropic"], 6)
    openai_usd = round(totals["openai"], 6)
    total_usd = round(anthropic_usd + openai_usd + embedding_usd, 6)

    return {
        "anthropic_usd": anthropic_usd,
        "openai_usd":    openai_usd,
        "embedding_usd": embedding_usd,
        "total_usd":     total_usd,
        "pricing_version": PRICING_VERSION,
        "by_phase":      by_phase,
    }
