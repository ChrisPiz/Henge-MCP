"""Frame → canonical model id mapping for Henge v0.6.

Distribution: 2× anthropic/sonnet-4-6 + 1× anthropic/opus-4-7 + 6× openai/gpt-5.
Rationale (from v0.6 spec):
- Empirical, first-principles, systemic, soft-contrarian, radical-optimist,
  pre-mortem → gpt-5 (numerical reasoning, multi-hop, contrarian reframe).
- Historical → opus-4-7 (best long-form recall of specific cases).
- Analogical, ethical → sonnet-4-6 (cross-domain mapping, deontological tension).

Canonical ids resolve through ``henge.providers.registry``; the raw SDK model
strings live inside each provider and never leak here.
"""
from __future__ import annotations

FRAME_MODEL_MAP: dict[str, str] = {
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


def model_for(frame_name: str) -> str:
    """Return the canonical model id for a frame. Raises KeyError if unknown."""
    return FRAME_MODEL_MAP[frame_name]
