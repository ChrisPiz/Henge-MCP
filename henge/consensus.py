"""Consensus synthesis — Haiku reads the 9 frames and extracts what they agree on.

Why: a reader needs to see consensus BEFORE the dissent. Otherwise the tenth-man
has no anchor to attack, narratively. Cost: ~CLP 30, +3-5s latency.
"""

HAIKU = "claude-haiku-4-5-20251001"
CONSENSUS_MAX_TOKENS = 800

CONSENSUS_SYSTEM = """Recibirás 9 análisis de una decisión, cada uno desde un ángulo cognitivo distinto (cada uno es un consejero). Tu trabajo: sintetizar el consenso emergente entre ellos.

FORMATO DE SALIDA (estricto):

1. **Empieza con un titular markdown h1 (`# ...`)** de 6–14 palabras que sintetice el ángulo del consenso para esta pregunta específica. Sin números de marco ni etiquetas de sistema. Ejemplo: "# Validar antes de contratar — el riesgo asimétrico domina".
2. **(1) Donde convergen los nueve** — usa exactamente este encabezado como `## (1) Donde convergen los nueve`. 1 párrafo describiendo los puntos en común.
3. **(2) Tensión interna** — usa `## (2) Tensión interna`. 1 párrafo sobre el desacuerdo o matiz que persiste entre los 9 (si hay alguno; si no hay, escribe "No hay tensión sustancial — todos los marcos apuntan al mismo lado.").
4. **(3) Inclinación neta** — usa `## (3) Inclinación neta`. 1 párrafo cerrando con la dirección que apunta el consenso. Empieza con `**Inclinación neta:**` en negrita.

Reglas de contenido:
- 2-4 puntos donde los 9 convergen (aunque cada uno los justifica distinto).
- NO listes los consejeros por nombre. NO cites a cada uno.
- Foco: qué creen los 9 EN COMÚN. Qué tendencia neta emerge.
- NO interpretes la decisión por el usuario. NO recomiendes nada nuevo. Solo sintetiza.

Responde en español. Total: titular + 3 secciones (~3 párrafos)."""


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
