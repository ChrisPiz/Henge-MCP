"""Meta-frame — audits the question itself before spending on the 9 advisors.

v0.6 design: gpt-5 [OpenAI] runs cross-lab audit on the question + canonical
context. If the question is exploration disguised as decision, or a proxy for
the real question, the meta-frame recommends reformulation and the server
short-circuits the run before the 9 frames fire. This saves ~$1.50 per
malformed run.

The full implementation of ``evaluate_question_quality`` lands in Task 4.2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_META_FRAME = _flag("HENGE_ENABLE_META_FRAME", True)


DecisionClass = Literal["reversible", "one-way-door", "two-way-with-cost", "unknown"]
Urgency = Literal["now", "weeks", "months", "fake-urgency", "unknown"]
QuestionQuality = Literal[
    "well-formed",
    "proxy-for-other-question",
    "exploration-disguised-as-decision",
    "unknown",
]
MetaRecommendation = Literal[
    "proceed",
    "reformulate",
    "postpone",
    "this-is-not-a-decision",
]


@dataclass
class MetaFrameResult:
    decision_class: DecisionClass
    urgency: Urgency
    question_quality: QuestionQuality
    suggested_reformulation: str | None
    meta_recommendation: MetaRecommendation
    reasoning: str
    gpt5_usage: dict | None


import json

from henge.providers import CompletionRequest, complete


_META_MODEL = "openai/gpt-5"
META_MAX_TOKENS = 4000  # gpt-5 reasoning headroom (~1500 internal + 2000 visible)


_META_SYSTEM = """You audit a decision question before 9 advisors spend tokens on it. You are NOT one of the advisors. Your job is to classify the question itself.

Classify along four axes:

1. decision_class: one of
   - "reversible": the user can undo this in <30 days at <10% of the cost
   - "one-way-door": effectively irreversible (closing a company, public commitment, contract signed)
   - "two-way-with-cost": reversible but costly (months and meaningful capital to undo)

2. urgency: one of
   - "now": material adverse outcome if not decided in days
   - "weeks": meaningful change-of-state in weeks
   - "months": months of runway before forced
   - "fake-urgency": the user feels urgency that the situation does not actually warrant

3. question_quality: one of
   - "well-formed": binary or small-N decision with clear stakes and stated criteria
   - "proxy-for-other-question": the named question is a stand-in for a deeper unspoken one
   - "exploration-disguised-as-decision": the user is still gathering information, not deciding

4. meta_recommendation: one of
   - "proceed": the 9 advisors should run as-is
   - "reformulate": offer the user a sharper version of the question first
   - "postpone": the user lacks information they could gather cheaply before deciding
   - "this-is-not-a-decision": the question is asking for explanation/exploration, not a commit

If meta_recommendation is "reformulate", set suggested_reformulation to a single-sentence rewrite of the question that would be well-formed.

LANGUAGE: write the `reasoning` and `suggested_reformulation` STRINGS in the SAME LANGUAGE as the question. If the question is in Spanish, write those two fields in Spanish. The classification VALUES (decision_class, urgency, question_quality, meta_recommendation) stay in English — they are enums consumed by code.

Output STRICT JSON. No prose. No markdown fence. Exact shape:
{
  "decision_class": "<one of above>",
  "urgency": "<one of above>",
  "question_quality": "<one of above>",
  "suggested_reformulation": "<sentence>" or null,
  "meta_recommendation": "<one of above>",
  "reasoning": "<2-3 paragraphs explaining the classifications>"
}
"""


_VALID_DECISION = {"reversible", "one-way-door", "two-way-with-cost"}
_VALID_URGENCY = {"now", "weeks", "months", "fake-urgency"}
_VALID_QUALITY = {"well-formed", "proxy-for-other-question", "exploration-disguised-as-decision"}
_VALID_RECOMMENDATION = {"proceed", "reformulate", "postpone", "this-is-not-a-decision"}


def _validate(value: str, allowed: set[str], fallback: str = "unknown") -> str:
    return value if value in allowed else fallback


async def evaluate_question_quality(
    question: str, canonical_context: str
) -> MetaFrameResult:
    """Audit the question itself with gpt-5 cross-lab. Returns a passthrough
    proceed result when the feature flag is off or the call/parse fails."""
    if not ENABLE_META_FRAME:
        return MetaFrameResult(
            decision_class="unknown",
            urgency="unknown",
            question_quality="unknown",
            suggested_reformulation=None,
            meta_recommendation="proceed",
            reasoning="meta-frame disabled by HENGE_ENABLE_META_FRAME=false",
            gpt5_usage=None,
        )

    user = (
        f"Question to audit:\n{question}\n\n"
        f"Canonical context (the user's situation, already cleaned up by Opus):\n{canonical_context}"
    )
    req = CompletionRequest(
        system=_META_SYSTEM,
        user=user,
        max_tokens=META_MAX_TOKENS,
        temperature=0.0,
        reasoning_effort="low",
    )
    try:
        resp = await complete(_META_MODEL, req)
    except Exception:
        return MetaFrameResult(
            decision_class="unknown",
            urgency="unknown",
            question_quality="unknown",
            suggested_reformulation=None,
            meta_recommendation="proceed",
            reasoning="meta-frame call failed; proceeding with the 9 advisors",
            gpt5_usage=None,
        )

    usage = {
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }

    text = resp.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return MetaFrameResult(
            decision_class="unknown",
            urgency="unknown",
            question_quality="unknown",
            suggested_reformulation=None,
            meta_recommendation="proceed",
            reasoning=resp.text[:500] or "could not parse meta-frame JSON",
            gpt5_usage=usage,
        )

    if not isinstance(parsed, dict):
        return MetaFrameResult(
            decision_class="unknown",
            urgency="unknown",
            question_quality="unknown",
            suggested_reformulation=None,
            meta_recommendation="proceed",
            reasoning="meta-frame returned non-object JSON",
            gpt5_usage=usage,
        )

    sr = parsed.get("suggested_reformulation")
    return MetaFrameResult(
        decision_class=_validate(str(parsed.get("decision_class", "")), _VALID_DECISION),
        urgency=_validate(str(parsed.get("urgency", "")), _VALID_URGENCY),
        question_quality=_validate(str(parsed.get("question_quality", "")), _VALID_QUALITY),
        suggested_reformulation=str(sr).strip() if sr else None,
        meta_recommendation=_validate(
            str(parsed.get("meta_recommendation", "")), _VALID_RECOMMENDATION, fallback="proceed"
        ),
        reasoning=str(parsed.get("reasoning", "")).strip(),
        gpt5_usage=usage,
    )
