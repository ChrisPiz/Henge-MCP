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
