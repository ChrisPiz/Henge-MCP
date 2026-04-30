"""Loads prompts at import, runs 9 cognitive frames in parallel + tenth-man dissenter sequentially."""
import asyncio
from pathlib import Path

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
    "tenth-man": "10-tenth-man.md",
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

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-7"
FRAME_MAX_TOKENS = 1500   # 4-6 paragraphs of reasoning per frame
TENTH_MAX_TOKENS = 3500   # 5-7 paragraphs for tenth-man (more cognitively demanding)


async def _call_anthropic(client, model, system, user, max_tokens):
    msg = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


async def run_agent(client, frame, question, context=None):
    """Run one cognitive frame. Returns response text."""
    system = PROMPTS[frame]
    user = question if not context else f"{question}\n\nContexto adicional:\n{context}"
    return await _call_anthropic(client, SONNET, system, user, FRAME_MAX_TOKENS)


async def run_tenth_man(client, successful_frames, question):
    """Receives only successful frame responses, returns steel-man dissent.

    successful_frames: list of (frame_name, response_text) tuples.
    """
    consensus_block = "\n\n".join(
        f"### Marco {i+1} — {frame}\n{resp}"
        for i, (frame, resp) in enumerate(successful_frames)
    )
    user = (
        f"Pregunta original:\n{question}\n\n"
        f"Los siguientes análisis convergen en un consenso. "
        f"Tu trabajo: asume que están TODOS equivocados y construye el contraargumento más fuerte y coherente posible. "
        f"Steel-man del disenso, no straw-man.\n\n"
        f"{consensus_block}"
    )
    return await _call_anthropic(client, OPUS, PROMPTS[TENTH_MAN], user, TENTH_MAX_TOKENS)


async def run_agents(client, question, context=None):
    """Run all 9 frames in parallel, then tenth-man.

    Returns: list of (frame_name, response_text, status) tuples, length 10.
    status is "ok" or "failed".

    Raises RuntimeError if fewer than 8/9 frames succeeded — without enough
    cognitive coverage, the dissenter has no real consensus to attack.
    """
    tasks = [run_agent(client, frame, question, context) for frame in FRAMES]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    nine = []
    for frame, res in zip(FRAMES, raw):
        if isinstance(res, BaseException):
            nine.append((frame, f"[failed: {type(res).__name__}: {res}]", "failed"))
        else:
            nine.append((frame, res, "ok"))

    n_ok = sum(1 for _, _, s in nine if s == "ok")
    if n_ok < 8:
        failed_frames = [f for f, _, s in nine if s == "failed"]
        raise RuntimeError(
            f"Solo {n_ok}/9 marcos cognitivos exitosos. Abortando (mínimo requerido: 8). "
            f"Marcos fallidos: {failed_frames}"
        )

    successful = [(f, r) for f, r, s in nine if s == "ok"]
    tenth_response = await run_tenth_man(client, successful, question)

    return nine + [(TENTH_MAN, tenth_response, "ok")]
