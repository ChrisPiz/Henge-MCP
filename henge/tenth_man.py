"""Tenth-man dual dissenter — Opus blind + gpt-5 informed.

v0.6 design:
  - Blind: Opus 4.7 sees only question + canonical_context, no 9 advisors.
    Produces the strongest pure dissent. Distance metric uses this.
  - Informed: gpt-5 sees the 9 + the blind, reconciles. Output:
      what_holds   — claims from blind that survive after seeing the 9
      what_revised — claims that get refined
      what_discarded — claims that turn out to be Opus bias

Cross-lab: blind on Anthropic, informed on OpenAI. If gpt-5 keeps most of
the blind, dissent is robust. If gpt-5 discards much, the blind was lab bias
masquerading as insight.

Implementation of ``run_tenth_man_blind`` and ``run_tenth_man_informed``
lands in Tasks 5.2 / 5.3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_TENTH_MAN_DUAL = _flag("HENGE_ENABLE_TENTH_MAN_DUAL", True)


@dataclass
class TenthManBlindResult:
    text: str
    opus_usage: dict | None


@dataclass
class TenthManInformedResult:
    text: str
    what_holds: list[str] = field(default_factory=list)
    what_revised: list[str] = field(default_factory=list)
    what_discarded: list[str] = field(default_factory=list)
    gpt5_usage: dict | None = None


import json
from pathlib import Path

from henge.providers import CompletionRequest, complete


_BLIND_MODEL = "anthropic/opus-4-7"
_INFORMED_MODEL = "openai/gpt-5"

BLIND_MAX_TOKENS = 6000   # Opus reasoning + 5-7 paragraphs of dissent
INFORMED_MAX_TOKENS = 4000  # gpt-5 reasoning_effort=low + structured JSON

# Reuse the canonical tenth-man prompt for the blind run. Loaded once at module import.
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_BLIND_SYSTEM = (_PROMPTS_DIR / "10-tenth-man-steelman.md").read_text(encoding="utf-8").strip()


_INFORMED_SYSTEM = """You audit a blind dissent against the 9 advisors that were just produced.

Your inputs:
- The original decision question.
- The canonical context (user's situation).
- The 9 advisors' outputs (their analyses).
- The blind tenth-man dissent (Opus 4.7, written WITHOUT seeing the 9).

Your job: as a cross-lab second opinion (gpt-5 from a different reasoning lineage), reconcile the blind dissent against the 9.

Decide:
- what_holds: claims from the blind that survive when you read them next to the 9.
- what_revised: claims that need refinement (the blind was directionally right but imprecise).
- what_discarded: claims that turn out to be model bias (not insight).

Then write a short refined dissent (2-4 paragraphs) that integrates only what holds + what was revised.

LANGUAGE: write the `text`, `what_holds`, `what_revised`, `what_discarded` IN THE SAME LANGUAGE as the question. If the question is in Spanish, write everything in Spanish.

Output STRICT JSON. No prose. No markdown fence. Exact shape:
{
  "text": "<refined dissent, 2-4 paragraphs>",
  "what_holds": ["<claim that holds>", ...],
  "what_revised": ["<claim that gets refined and how>", ...],
  "what_discarded": ["<claim that turns out to be lab bias>", ...]
}
"""


def _usage_dict(resp) -> dict:
    return {
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }


def _strip_md_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


async def run_tenth_man_blind(
    question: str, canonical_context: str
) -> TenthManBlindResult:
    """Opus blind dissent. Sees only question + context, NOT the 9 advisors.

    Anticipates whatever consensus the 9 might converge on and writes the
    strongest possible counter-argument. The distance metric of the report
    uses this output's embedding.
    """
    user = (
        f"Original question:\n{question}\n\n"
        f"User context (canonical):\n{canonical_context}\n\n"
        f"Note: no consensus has been generated yet. Anticipate any plausible "
        f"consensus the 9 advisors might converge on for this question, then "
        f"build the strongest steel-man dissent possible. Do not address "
        f"specific advisors — you have not seen them."
    )
    req = CompletionRequest(
        system=_BLIND_SYSTEM,
        user=user,
        max_tokens=BLIND_MAX_TOKENS,
        temperature=0.0,
    )
    try:
        resp = await complete(_BLIND_MODEL, req)
    except Exception as exc:
        return TenthManBlindResult(
            text=f"[failed: blind dissent call errored: {type(exc).__name__}]",
            opus_usage=None,
        )

    return TenthManBlindResult(
        text=resp.text.strip(),
        opus_usage=_usage_dict(resp),
    )


async def run_tenth_man_informed(
    question: str,
    canonical_context: str,
    nine_outputs: list[tuple[str, str]],
    blind_text: str,
) -> TenthManInformedResult:
    """gpt-5 informed reconciliation. Sees the 9 + the blind, returns structured diff."""
    nine_block = "\n\n".join(
        f"### Advisor {i+1} — {frame}\n{text}"
        for i, (frame, text) in enumerate(nine_outputs)
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"Canonical context:\n{canonical_context}\n\n"
        f"=== The 9 advisors ===\n{nine_block}\n\n"
        f"=== Blind tenth-man dissent (Opus, written without seeing the 9) ===\n{blind_text}"
    )
    req = CompletionRequest(
        system=_INFORMED_SYSTEM,
        user=user,
        max_tokens=INFORMED_MAX_TOKENS,
        temperature=0.0,
        reasoning_effort="low",
    )
    try:
        resp = await complete(_INFORMED_MODEL, req)
    except Exception as exc:
        return TenthManInformedResult(
            text=f"[failed: informed dissent call errored: {type(exc).__name__}]",
            gpt5_usage=None,
        )

    usage = _usage_dict(resp)
    raw = _strip_md_fence(resp.text)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return TenthManInformedResult(
            text=resp.text.strip(),
            gpt5_usage=usage,
        )

    if not isinstance(parsed, dict):
        return TenthManInformedResult(
            text=resp.text.strip(),
            gpt5_usage=usage,
        )

    def _str_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    return TenthManInformedResult(
        text=str(parsed.get("text", "")).strip(),
        what_holds=_str_list(parsed.get("what_holds")),
        what_revised=_str_list(parsed.get("what_revised")),
        what_discarded=_str_list(parsed.get("what_discarded")),
        gpt5_usage=usage,
    )
