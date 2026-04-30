"""Scoping phase — generates clarifying questions before running the 10 agents.

Without scoping, frames speculate or stay generic. With scoping, the user provides
the missing context (income, location, constraints) so frames apply to facts.

Uses Haiku for speed/cost: 1 call ≈ CLP 10, returns in ~3-5s.
"""
import json

HAIKU = "claude-haiku-4-5-20251001"
SCOPING_MAX_TOKENS = 800

SCOPING_SYSTEM = """Recibirás una pregunta de decisión. Tu trabajo: generar 4-7 preguntas concretas que un asesor experto haría al usuario antes de poder dar consejo fundado.

Las respuestas alimentarán a 9 marcos cognitivos: empírico (datos, números), histórico (precedentes, casos), primer-principios (restricciones), analógico, sistémico (segundo orden), ético (stakeholders), contrarian (supuestos), pre-mortem (modos de falla), optimista (upside).

Busca preguntas que cubran (cuando apliquen al dominio):
- Datos cuantitativos personales (ingresos, ahorros, plazos, deudas, edad, dependientes)
- Restricciones y deal-breakers
- Geografía, comunidad, ubicación
- Relaciones y stakeholders afectados
- Preferencias subjetivas, filosofía de vida, prioridades
- Información que NO esté ya en la pregunta original

Reglas:
- 4-7 preguntas, no más, no menos.
- Cada una concreta, específica al dominio. NO genérica.
- Una pregunta por entry — sin "y" compuestas.
- NO repitas información ya en la pregunta.
- En español.

Output: JSON array de strings. SOLO el JSON, sin prosa, sin markdown fence.
Ejemplo formato: ["¿Cuál es tu ingreso mensual neto aproximado?", "¿En qué comunas estarías dispuesto a vivir?"]"""


async def generate_questions(client, question):
    """Returns list of 4-7 clarifying questions, or None on failure.

    Tolerates 3-8 questions to handle edge cases. Strips markdown code fences
    that Haiku sometimes wraps around JSON.
    """
    try:
        msg = await client.messages.create(
            model=HAIKU,
            max_tokens=SCOPING_MAX_TOKENS,
            system=SCOPING_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        text = msg.content[0].text.strip()
    except Exception:
        return None

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        questions = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(questions, list):
        return None
    if not (3 <= len(questions) <= 8):
        return None
    return [str(q).strip() for q in questions if str(q).strip()]
