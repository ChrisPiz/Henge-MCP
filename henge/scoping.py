"""Scoping phase — multi-step pipeline.

v0.6 pipeline:
  1. Haiku 4.5 [Anthropic] generates 4-7 base scoping questions.
  2. gpt-5 [OpenAI] adversarial review: 2-4 additional questions that
     challenge unexamined assumptions in the original question.
  3. Combined set returns to user.
  4. User responds (handled by server.py).
  5. Opus 4.7 [Anthropic] canonicalizes the user's responses into a
     summary + flags any inconsistencies.

Feature flags (env, default ``true``):
  - ``HENGE_ENABLE_ADVERSARIAL_SCOPING`` — skip step 2 if set to ``false``.
  - ``HENGE_ENABLE_CANONICAL_CONTEXT`` — skip step 5 if set to ``false``.

Cross-lab rationale: Haiku and Opus are Anthropic; gpt-5 is OpenAI. If Haiku
has a systematic bias about what counts as a "good scoping question", gpt-5
catches it from a different reasoning lineage.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal


# Feature flags read at import time. Override via .env or env vars.
def _flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name, "true" if default else "false").strip().lower()
    return val in ("1", "true", "yes", "on")


ENABLE_ADVERSARIAL = _flag("HENGE_ENABLE_ADVERSARIAL_SCOPING", True)
ENABLE_CANONICAL_CONTEXT = _flag("HENGE_ENABLE_CANONICAL_CONTEXT", True)


@dataclass
class ScopedQuestion:
    id: str
    text: str
    source: Literal["scoping", "adversarial"]
    challenges_assumption: str | None


@dataclass
class ScopingResult:
    questions: list[ScopedQuestion]
    adversarial_count: int
    version: str
    haiku_usage: dict | None
    gpt5_usage: dict | None


@dataclass
class CanonicalContext:
    summary: str
    flags: list[str] = field(default_factory=list)
    opus_usage: dict | None = None


# Pipeline functions (run_scoping, finalize_context) are added in Tasks 3.2 / 3.3.
# Back-compat shim ``generate_questions(client, question)`` lives at the bottom
# (added in Task 3.4) so server.py keeps booting until its call-sites migrate.


from henge.providers import CompletionRequest, complete


_HAIKU_MODEL = "anthropic/haiku-4-5"
_GPT5_MODEL = "openai/gpt-5"
_OPUS_MODEL = "anthropic/opus-4-7"

SCOPING_MAX_TOKENS = 800
ADVERSARIAL_MAX_TOKENS = 600
CANONICAL_MAX_TOKENS = 1200


_SCOPING_SYSTEM = """You will receive a decision question. Your job: generate 4-7 concrete questions an expert advisor would ask the user before being able to give grounded advice.

The answers will feed 9 advisors with distinct cognitive angles: empirical (data, numbers), historical (precedents, cases), first-principles (constraints), analogical, systemic (second-order), ethical (stakeholders), contrarian (assumptions), pre-mortem (failure modes), optimist (upside).

Look for questions that cover (when relevant to the domain):
- Personal quantitative data (income, savings, deadlines, debts, age, dependents)
- Constraints and deal-breakers
- Geography, community, location
- Relationships and affected stakeholders
- Subjective preferences, life philosophy, priorities
- Information NOT already in the original question

Rules:
- 4-7 questions, no more, no less.
- Each concrete, specific to the domain. NOT generic.
- One question per entry — no "and"-compound questions.
- DO NOT repeat information already in the question.
- Match the language of the original question.

Output: JSON array of strings. ONLY the JSON, no prose, no markdown fence.
Format example: ["What is your approximate net monthly income?", "Which neighborhoods would you be willing to live in?"]"""


_ADVERSARIAL_SYSTEM = """You audit scoping questions from a different reasoning school.

Inputs you receive:
1. The user's original decision question.
2. A set of base scoping questions another model generated.

Your job: identify 2-4 ADDITIONAL questions that challenge an unexamined assumption in the original question OR fill a gap the base set missed. Each adversarial question must:
- Surface an assumption hidden inside the question itself (not just request more data).
- Be answerable by the user (not philosophical).
- Avoid duplicating any base question.

Match the language of the original question.

Output: JSON array of objects with this exact shape, no prose, no markdown fence:
[
  {"text": "What if you are not actually trying to optimize for X?", "challenges_assumption": "that X is the right objective"},
  ...
]"""


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


async def _haiku_initial_scoping(question: str) -> tuple[list[str], dict | None]:
    """Returns (base_question_strings, usage). Empty list on parse failure."""
    req = CompletionRequest(
        system=_SCOPING_SYSTEM,
        user=question,
        max_tokens=SCOPING_MAX_TOKENS,
        temperature=0.0,
    )
    try:
        resp = await complete(_HAIKU_MODEL, req)
    except Exception:
        return [], None

    text = _strip_md_fence(resp.text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], _usage_dict(resp)

    if not isinstance(parsed, list):
        return [], _usage_dict(resp)
    if not (3 <= len(parsed) <= 8):
        return [], _usage_dict(resp)
    return [str(q).strip() for q in parsed if str(q).strip()], _usage_dict(resp)


async def _gpt5_adversarial_review(
    question: str, base_questions: list[str]
) -> tuple[list[dict], dict | None]:
    """Returns (adversarial_objects, usage). Empty list on parse failure."""
    user = (
        f"Original question:\n{question}\n\n"
        f"Base scoping questions already generated:\n"
        + "\n".join(f"- {q}" for q in base_questions)
    )
    req = CompletionRequest(
        system=_ADVERSARIAL_SYSTEM,
        user=user,
        max_tokens=ADVERSARIAL_MAX_TOKENS,
        temperature=0.0,
    )
    try:
        resp = await complete(_GPT5_MODEL, req)
    except Exception:
        return [], None

    text = _strip_md_fence(resp.text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], _usage_dict(resp)

    if not isinstance(parsed, list):
        return [], _usage_dict(resp)
    cleaned = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text_val = str(item.get("text", "")).strip()
        if not text_val:
            continue
        challenges = item.get("challenges_assumption")
        cleaned.append({
            "text": text_val,
            "challenges_assumption": str(challenges).strip() if challenges else None,
        })
    return cleaned, _usage_dict(resp)


async def run_scoping(question: str) -> ScopingResult:
    """Multi-step scoping: Haiku base + (optional) gpt-5 adversarial review."""
    base_strings, haiku_usage = await _haiku_initial_scoping(question)

    questions: list[ScopedQuestion] = [
        ScopedQuestion(
            id=f"q_{i+1:03d}",
            text=q,
            source="scoping",
            challenges_assumption=None,
        )
        for i, q in enumerate(base_strings)
    ]

    gpt5_usage: dict | None = None
    adv_objs: list[dict] = []
    if ENABLE_ADVERSARIAL:
        adv_objs, gpt5_usage = await _gpt5_adversarial_review(question, base_strings)
        offset = len(questions)
        for j, obj in enumerate(adv_objs):
            questions.append(
                ScopedQuestion(
                    id=f"q_{offset + j + 1:03d}",
                    text=obj["text"],
                    source="adversarial",
                    challenges_assumption=obj.get("challenges_assumption"),
                )
            )

    return ScopingResult(
        questions=questions,
        adversarial_count=len(adv_objs),
        version="v0.6",
        haiku_usage=haiku_usage,
        gpt5_usage=gpt5_usage,
    )


_CANONICAL_SYSTEM = """You receive an original decision question and the user's free-form context (their answers to scoping questions, possibly unstructured).

Produce a tight executive summary suitable for feeding into 9 cognitive advisors. Goals:
- Preserve every quantitative detail the user gave (numbers, dates, names, locations).
- Resolve ambiguous references when possible.
- Flag any internal inconsistencies the user wrote.

Output format: plain prose, 2-4 paragraphs maximum. No headers, no bullets.

After the summary, if you found inconsistencies, append a single line:
INCONSISTENCIES: <semicolon-separated short flags>

Match the language of the original question."""


async def finalize_context(question: str, context: str) -> CanonicalContext:
    """Opus canonicalizes the user's free-form context into a summary + flags.

    When ``HENGE_ENABLE_CANONICAL_CONTEXT`` is false, the function passes the
    context through unchanged with empty flags and ``opus_usage=None``.
    """
    if not ENABLE_CANONICAL_CONTEXT:
        return CanonicalContext(summary=context, flags=[], opus_usage=None)

    user = f"Original question:\n{question}\n\nUser context:\n{context}"
    req = CompletionRequest(
        system=_CANONICAL_SYSTEM,
        user=user,
        max_tokens=CANONICAL_MAX_TOKENS,
        temperature=0.0,
    )
    try:
        resp = await complete(_OPUS_MODEL, req)
    except Exception:
        return CanonicalContext(summary=context, flags=[], opus_usage=None)

    text = resp.text.strip()
    flags: list[str] = []
    if "INCONSISTENCIES:" in text:
        head, _, tail = text.partition("INCONSISTENCIES:")
        text = head.strip()
        flags = [f.strip() for f in tail.split(";") if f.strip()]

    return CanonicalContext(
        summary=text,
        flags=flags,
        opus_usage=_usage_dict(resp),
    )
