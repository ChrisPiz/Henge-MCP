"""Loads prompts at import; runs the 9 frames in parallel with the Opus blind
dissenter, then the gpt-5 informed dissenter sequentially. v0.6 dual tenth-man.
"""
import asyncio
import hashlib
from pathlib import Path

from henge.config.frame_assignment import FRAME_MODEL_MAP, model_for
from henge.providers import CompletionRequest, complete
from henge.tenth_man import (
    TenthManBlindResult,
    TenthManInformedResult,
    run_tenth_man_blind,
    run_tenth_man_informed,
)

FRAMES = [
    "empirical",
    "historical",
    "first-principles",
    "analogical",
    "systemic",
    "ethical",
    "soft-contrarian",
    "radical-optimist",
    "pre-mortem",
]
TENTH_MAN = "tenth-man"

_FILE_MAP = {
    "empirical": "01-empirical.md",
    "historical": "02-historical.md",
    "first-principles": "03-first-principles.md",
    "analogical": "04-analogical.md",
    "systemic": "05-systemic.md",
    "ethical": "06-ethical.md",
    "soft-contrarian": "07-soft-contrarian.md",
    "radical-optimist": "08-radical-optimist.md",
    "pre-mortem": "09-pre-mortem.md",
    "tenth-man": "10-tenth-man-steelman.md",
}

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompts() -> dict:
    """Load all prompt markdowns once at module import. Single source of truth.

    Loaded eagerly so disk reads never happen on the hot path, and prompts
    cannot drift between concurrent invocations.
    """
    prompts = {}
    for name, filename in _FILE_MAP.items():
        path = _PROMPT_DIR / filename
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Prompt file empty: {filename}")
        prompts[name] = text
    return prompts


PROMPTS = _load_prompts()


def _compute_prompts_hash() -> str:
    """SHA256 over ordered concat of the 10 prompt files. Short prefix for readability.

    Persisted in every report so any future change to prompts is traceable —
    reports with a different hash are not directly comparable.
    """
    blob = "".join(PROMPTS[name] for name in [*FRAMES, TENTH_MAN]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


PROMPTS_HASH = _compute_prompts_hash()

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-7"
# gpt-5 (used by 6 of 9 frames) is a reasoning model that burns ~1500+
# tokens internally on chain-of-thought before producing visible output.
# Below ~2500 the visible content can be empty (finish_reason='length').
# 4000 leaves headroom for reasoning + 4-6 paragraphs of visible content.
# Same reasoning applies to TENTH_MAX_TOKENS — Opus 4.7 is also reasoning.
FRAME_MAX_TOKENS = 4000   # 4-6 paragraphs of reasoning per frame
TENTH_MAX_TOKENS = 6000   # 5-7 paragraphs for tenth-man (more cognitively demanding)

# v0.5: temperature=0 pinned across all calls. Reproducibility > stylistic
# variance — see WHITEPAPER.md "Pre-registered runtime decisions".
TEMPERATURE = 0

# Reasoning-tier models (e.g. Opus 4.7 with extended thinking enabled by
# default) reject the ``temperature`` parameter outright. We omit it for
# those models and rely on the model's built-in determinism for the
# tenth-man dissent. Sonnet 4.6 and Haiku 4.5 still accept temperature=0.
# Add models to this set as Anthropic ships new reasoning models that
# refuse temperature; the runtime fallback below will also catch unknown
# cases gracefully.
MODELS_WITHOUT_TEMPERATURE = {OPUS}


def _extract_usage(msg, model):
    """Read token counts from an Anthropic message. Tolerant to mock objects.

    Real Anthropic responses always populate ``usage``. Tests may stub messages
    without it; in that case we record zeros so cost accounting still works.
    """
    usage = getattr(msg, "usage", None)
    return {
        "model": model,
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


async def _call_anthropic(client, model, system, user, max_tokens, temperature=TEMPERATURE):
    """Run a single Anthropic call. Returns ``(text, usage_dict)``.

    Usage dict: ``{"model": ..., "input_tokens": int, "output_tokens": int}``.

    ``temperature`` defaults to the pre-registered ``TEMPERATURE = 0`` so any
    caller that does not opt in still gets reproducible runs. K-runs mode
    (server.decide(k_runs=K, run_temperature=T)) overrides per call.

    Models in ``MODELS_WITHOUT_TEMPERATURE`` reject the parameter (reasoning-
    tier with mandatory extended thinking) — we omit it for those. Any other
    model that surfaces a temperature-related API error triggers a one-shot
    retry without the parameter so we degrade gracefully when Anthropic adds
    new reasoning models without us shipping a code update.
    """
    base_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    if model in MODELS_WITHOUT_TEMPERATURE:
        msg = await client.messages.create(**base_kwargs)
    else:
        try:
            msg = await client.messages.create(**base_kwargs, temperature=temperature)
        except Exception as exc:
            err = str(exc).lower()
            temperature_rejected = any(
                kw in err
                for kw in ("temperature", "extended thinking", "thinking is enabled")
            )
            if not temperature_rejected:
                raise
            msg = await client.messages.create(**base_kwargs)

    return msg.content[0].text, _extract_usage(msg, model)


async def run_agent(frame, question, context=None, temperature=TEMPERATURE):
    """Run one cognitive frame via its assigned model. Returns (text, usage_dict).

    The frame→model mapping comes from ``henge.config.frame_assignment``.
    Each canonical id (e.g. ``openai/gpt-5``) is resolved through the
    provider registry; the raw SDK model string never appears here.
    """
    model_id = model_for(frame)
    system = PROMPTS[frame]
    user = question if not context else f"{question}\n\nAdditional context:\n{context}"
    # "low" reasoning effort: gpt-5 frames otherwise burn the entire
    # max_tokens budget on internal chain-of-thought, returning empty
    # content. Anthropic providers ignore this kwarg.
    req = CompletionRequest(
        system=system,
        user=user,
        max_tokens=FRAME_MAX_TOKENS,
        temperature=temperature,
        reasoning_effort="low",
    )
    resp = await complete(model_id, req)
    usage = {
        "model": model_id,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
    }
    return resp.text, usage


# Legacy v0.5 single-tenth-man path. Replaced in Phase 5 by run_tenth_man_blind
# (Opus, sees no advisors) + run_tenth_man_informed (gpt-5, sees the 9 + blind).
# Kept here as dead code until Phase 8 schema cleanup; do not call from run_agents.
async def run_tenth_man(client, successful_frames, question, temperature=TEMPERATURE):
    """Receives only successful frame responses, returns ``(text, usage_dict)``.

    successful_frames: list of (frame_name, response_text) tuples.
    """
    consensus_block = "\n\n".join(
        f"### Advisor {i+1} — {frame}\n{resp}"
        for i, (frame, resp) in enumerate(successful_frames)
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"The following analyses converge on a consensus. "
        f"Your job: assume they are ALL wrong and build the strongest, most coherent counter-argument possible. "
        f"Steel-man the dissent, never straw-man.\n\n"
        f"{consensus_block}"
    )
    return await _call_anthropic(client, OPUS, PROMPTS[TENTH_MAN], user, TENTH_MAX_TOKENS, temperature)


async def run_agents(question, context=None, temperature=TEMPERATURE):
    """Run 9 frames in parallel with the blind tenth-man, then the informed
    tenth-man sequentially.

    Returns a dict::

        {
          "nine": list of (frame_name, response_text, status, usage_dict),
          "blind": TenthManBlindResult,
          "informed": TenthManInformedResult,
        }

    ``status`` is "ok" or "failed". On "failed", usage_dict is None.

    Raises RuntimeError if fewer than 8/9 frames succeeded — without enough
    cognitive coverage, the dissenter has no real consensus to attack.

    The blind runs in parallel with the 9 (no dependency). Informed runs
    sequentially after both finish (it sees the 9 + blind).
    """
    nine_coros = [run_agent(frame, question, context, temperature) for frame in FRAMES]
    blind_coro = run_tenth_man_blind(question, context or "")

    # Run nine + blind concurrently. Use asyncio.gather over a unified list
    # so they actually run in parallel (rather than sequentially awaiting).
    all_coros = nine_coros + [blind_coro]
    raw = await asyncio.gather(*all_coros, return_exceptions=True)

    nine_raw = raw[:9]
    blind_raw = raw[9]

    nine = []
    for frame, res in zip(FRAMES, nine_raw):
        if isinstance(res, BaseException):
            nine.append((frame, f"[failed: {type(res).__name__}: {res}]", "failed", None))
        else:
            text, usage = res
            nine.append((frame, text, "ok", usage))

    n_ok = sum(1 for _, _, s, _ in nine if s == "ok")
    if n_ok < 8:
        failed_frames = [f for f, _, s, _ in nine if s == "failed"]
        raise RuntimeError(
            f"Only {n_ok}/9 advisors succeeded. Aborting (minimum required: 8). "
            f"Failed advisors: {failed_frames}"
        )

    if isinstance(blind_raw, BaseException):
        blind = TenthManBlindResult(
            text=f"[failed: blind dissent crashed: {type(blind_raw).__name__}]",
            opus_usage=None,
        )
    else:
        blind = blind_raw

    successful = [(f, r) for f, r, s, _ in nine if s == "ok"]
    informed = await run_tenth_man_informed(question, context or "", successful, blind.text)

    return {"nine": nine, "blind": blind, "informed": informed}
