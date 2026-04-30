"""Consensus synthesis — Haiku reads the 9 frames and extracts what they agree on.

Why: a reader needs to see consensus BEFORE the dissent. Otherwise the tenth-man
has no anchor to attack, narratively. Cost: ~CLP 30, +3-5s latency.
"""

HAIKU = "claude-haiku-4-5-20251001"
CONSENSUS_MAX_TOKENS = 800

CONSENSUS_SYSTEM = """Recibirás 9 análisis de una decisión, cada uno desde un ángulo cognitivo distinto (cada uno es un consejero). Tu trabajo: sintetizar el consenso emergente entre ellos.

Reglas:
1. Identifica los 2-4 puntos donde los nueve consejeros convergen (aunque cada uno los justifica de forma distinta).
2. Resume en 2-3 párrafos compactos. NO listes los consejeros por nombre. NO cites a cada uno.
3. Foco: qué creen los 9 EN COMÚN sobre la decisión. Qué tendencia neta emerge.
4. Cierra con una sola línea destacada (en su propio párrafo): "**Inclinación neta:** ..." con la dirección que apunta el consenso.

NO interpretes la decisión por el usuario. NO recomiendes nada nuevo que no esté ya en los 9. Solo sintetiza lo que ya está.

Responde en español. 2-3 párrafos + 1 línea de cierre."""


async def synthesize_consensus(client, frames_responses, question):
    """Synthesize consensus from the 9 frame responses. Returns text or None on failure.

    Args:
        client: AsyncAnthropic instance.
        frames_responses: list of (frame_name, response_text) tuples.
        question: original user question.
    """
    if not frames_responses:
        return None

    block = "\n\n".join(
        f"### {name}\n{resp}" for name, resp in frames_responses
    )
    user = f"Pregunta original:\n{question}\n\n9 análisis:\n\n{block}"

    try:
        msg = await client.messages.create(
            model=HAIKU,
            max_tokens=CONSENSUS_MAX_TOKENS,
            system=CONSENSUS_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None
