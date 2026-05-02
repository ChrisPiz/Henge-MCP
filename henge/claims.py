"""Claim extraction + cross-lab verification.

v0.6 design:
  1. Sonnet 4.6 reads the Haiku-synthesized consensus and extracts N
     falsifiable claims, each tagged factual / prescriptive / causal.
  2. gpt-5 [OpenAI, cross-lab] reads each claim against the 9 frame outputs
     and labels it: which frames support it, which contest it, and how
     strong the support is.

Without this layer, the consensus could hallucinate a claim (the synth
lab's bias) that no advisor actually said. Cross-lab verification catches
those because gpt-5 has no affinity with Haiku's output style.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

from henge.providers import CompletionRequest, complete


def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_CLAIM_VERIFICATION = _flag("HENGE_ENABLE_CLAIM_VERIFICATION", True)


ClaimType = Literal["factual", "prescriptive", "causal", "unknown"]
SupportStrength = Literal["strong", "moderate", "weak", "unsupported"]


@dataclass
class Claim:
    text: str
    type: ClaimType


@dataclass
class ClaimVerification:
    claim_text: str
    claim_type: str
    supporting_frames: list[str] = field(default_factory=list)
    contesting_frames: list[str] = field(default_factory=list)
    support_strength: SupportStrength = "unsupported"


_EXTRACT_MODEL = "anthropic/sonnet-4-6"
_VERIFY_MODEL = "openai/gpt-5"

EXTRACT_MAX_TOKENS = 2000
VERIFY_MAX_TOKENS = 4000


_EXTRACT_SYSTEM = """You read a synthesized consensus statement and extract its falsifiable claims.

A falsifiable claim is a single proposition that can be checked against evidence — not a vague gesture. Tag each claim:
  - "factual": a statement of fact / state of the world (numbers, dates, named entities)
  - "prescriptive": a recommendation or normative statement (should, must, ought)
  - "causal": a claim about cause/effect or mechanism

Do not invent claims that aren't in the text. Do not skip claims that are. Aim for 4-10 claims that together capture the consensus's substance.

LANGUAGE: write each claim's `text` in the SAME LANGUAGE as the input consensus.

Output STRICT JSON. No prose. No markdown fence. Exact shape:
[
  {"text": "<claim>", "type": "factual" | "prescriptive" | "causal"},
  ...
]
"""


_VERIFY_SYSTEM = """You audit a list of claims (from a synthesized consensus) against the 9 advisor analyses they came from.

For each claim, decide:
  - supporting_frames: which advisors (by name) support the claim — explicitly or strongly implicitly.
  - contesting_frames: which advisors (by name) contest or contradict the claim.
  - support_strength:
      "strong"     — at least 6/9 advisors support the claim with no contradiction
      "moderate"   — 3-5/9 support, possibly minor contradiction
      "weak"       — 1-2/9 support
      "unsupported" — 0/9 support (the consensus hallucinated this claim)

Use the exact frame names from the 9 input advisors. Do not invent frames.

LANGUAGE: keep `claim_text` exactly as input. Other fields are enums (English) consumed by code.

Output STRICT JSON. No prose. No markdown fence. Exact shape:
[
  {
    "claim_text": "<exact text from input>",
    "claim_type": "<factual|prescriptive|causal>",
    "supporting_frames": ["<frame name>", ...],
    "contesting_frames": ["<frame name>", ...],
    "support_strength": "<strong|moderate|weak|unsupported>"
  },
  ...
]
"""


def _strip_md_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _usage_dict(resp) -> dict:
    return {
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }


def _validate_claim_type(value: str) -> str:
    return value if value in {"factual", "prescriptive", "causal"} else "unknown"


def _validate_strength(value: str) -> str:
    return value if value in {"strong", "moderate", "weak", "unsupported"} else "unsupported"


async def extract_claims(consensus_text: str) -> tuple[list[Claim], dict | None]:
    """Sonnet extracts falsifiable claims from the consensus. Returns
    (list, usage). Empty list on flag-off, parse failure, or call error."""
    if not ENABLE_CLAIM_VERIFICATION:
        return [], None
    if not consensus_text or not consensus_text.strip():
        return [], None

    req = CompletionRequest(
        system=_EXTRACT_SYSTEM,
        user=consensus_text,
        max_tokens=EXTRACT_MAX_TOKENS,
        temperature=0.0,
    )
    try:
        resp = await complete(_EXTRACT_MODEL, req)
    except Exception:
        return [], None

    usage = _usage_dict(resp)
    text = _strip_md_fence(resp.text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], usage

    if not isinstance(parsed, list):
        return [], usage

    claims: list[Claim] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        ct = str(item.get("text", "")).strip()
        if not ct:
            continue
        tp = _validate_claim_type(str(item.get("type", "")))
        claims.append(Claim(text=ct, type=tp))

    return claims, usage


async def verify_claims(
    claims: list[Claim],
    nine_outputs: list[tuple[str, str]],
) -> tuple[list[ClaimVerification], dict | None]:
    """gpt-5 cross-lab verifies each claim against the 9 frame outputs.
    Returns (list, usage). Empty list on flag-off, no input, parse failure,
    or call error."""
    if not ENABLE_CLAIM_VERIFICATION:
        return [], None
    if not claims:
        return [], None

    nine_block = "\n\n".join(
        f"### Advisor {i+1} — {frame}\n{text}"
        for i, (frame, text) in enumerate(nine_outputs)
    )
    claims_block = "\n".join(
        f"- ({c.type}) {c.text}" for c in claims
    )
    user = (
        f"Claims to verify:\n{claims_block}\n\n"
        f"=== The 9 advisor analyses ===\n{nine_block}"
    )
    req = CompletionRequest(
        system=_VERIFY_SYSTEM,
        user=user,
        max_tokens=VERIFY_MAX_TOKENS,
        temperature=0.0,
        reasoning_effort="low",
    )
    try:
        resp = await complete(_VERIFY_MODEL, req)
    except Exception:
        return [], None

    usage = _usage_dict(resp)
    text = _strip_md_fence(resp.text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], usage

    if not isinstance(parsed, list):
        return [], usage

    def _str_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    verifications: list[ClaimVerification] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        ct = str(item.get("claim_text", "")).strip()
        if not ct:
            continue
        verifications.append(
            ClaimVerification(
                claim_text=ct,
                claim_type=_validate_claim_type(str(item.get("claim_type", ""))),
                supporting_frames=_str_list(item.get("supporting_frames")),
                contesting_frames=_str_list(item.get("contesting_frames")),
                support_strength=_validate_strength(
                    str(item.get("support_strength", "unsupported"))
                ),
            )
        )

    return verifications, usage
