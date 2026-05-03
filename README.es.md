# Henge

> **Léelo en:** [English](README.md) · [Español](README.es.md)

**Structured Dissent Protocol** — 9 marcos cognitivos + 1 disidente
obligatorio construido por *steel-manning*, expuesto como servidor MCP.

![Henge](docs/header-v2.jpg)

Henge corre tu decisión a través de nueve lentes cognitivas en paralelo y
luego obliga a un décimo asesor — el **Décimo Hombre** — a construir el
caso *steel-man* más fuerte posible contra cualquier consenso al que lleguen
los nueve. Persiste un registro JSON + un reporte HTML editorial en disco y
expone un veredicto tri-estado guiado por el **Consensus Fragility Index
(CFI)**.

**Reclamo (v0.6.0).** Dado un protocolo pre-registrado con `temperature=0`,
la misma pregunta produce el mismo bin de CFI a través de las corridas,
módulo el ruido numérico de la inicialización de MDS. v0.6 divide la
corrida entre dos labs (Anthropic + OpenAI) por diseño, así las
alucinaciones del sintetizador las atrapa el auditor en lugar de que se
laven dentro del consenso. La validación contra calidad de decisión
(benchmark Henge-50) sigue pendiente.

[Demo](https://chrispiz.github.io/Henge-MCP/demo.html) · [Paper](WHITEPAPER.md) ·
[Limits](LIMITS.md) · [Methodology](METHODOLOGY.md) · [CFI spec](docs/cfi-spec.md) ·
[Manifesto](MANIFESTO.md) · [Developer](DEVELOPER.md)

> **Nota:** los documentos enlazados arriba (whitepaper, límites,
> metodología, spec de CFI, manifiesto, guía de desarrollo) están en inglés.
> Esta versión en español cubre el README; las traducciones de los demás
> docs están en backlog.

---

## Qué hace

```
pregunta
   ↓
┌─ fase 1 · scoping ─────────────────────────────────────┐
│ preguntas base       (Haiku 4.5)                       │
│ barrido adversarial  (gpt-5, cross-lab)                │
│ → 4–7 preguntas, 2–4 desafiando supuestos ocultos      │
│   dentro de la pregunta misma                          │
└────────────────────────────────────────────────────────┘
   ↓ respuestas del usuario
┌─ fase 2 · meta-frame ──────────────────────────────────┐
│ clasificar (gpt-5, cross-lab)                          │
│ → decision_class · urgency · question_quality          │
│   · meta_recommendation                                │
│ si proxy / exploración / urgencia falsa:               │
│   corto-circuito con suggested_reformulation           │
│   (ahorra ~USD 1.00/corrida)                           │
└────────────────────────────────────────────────────────┘
   ↓
┌─ fase 3 · contexto canónico ───────────────────────────┐
│ canonicaliza respuestas  (Opus 4.7)                    │
│ → resumen ejecutivo apretado + inconsistencias         │
│   marcadas, mostrado a los 9 asesores                  │
└────────────────────────────────────────────────────────┘
   ↓
┌─ fase 4 · 9 marcos en paralelo ────────────────────────┐
│ 6× gpt-5 + 2× Sonnet 4.6 + 1× Opus 4.7                 │
│ ↓                                                      │
│ embeddings (text-embedding-3-large)                    │
│ ↓                                                      │
│ MDS clásico + coseno                                   │
└────────────────────────────────────────────────────────┘
   ↓
┌─ fase 5 · síntesis + disenso dual ─────────────────────┐
│ consenso             (Haiku 4.5)                       │
│ décimo hombre ciego  (Opus 4.7, sin ver a los 9)       │
│ décimo hombre informado (gpt-5, cross-lab — ve los 9   │
│                          + ciego, devuelve what_holds /│
│                          what_revised / what_discarded)│
└────────────────────────────────────────────────────────┘
   ↓
┌─ fase 6 · verificación de claims ──────────────────────┐
│ extrae claims        (Sonnet 4.6)                      │
│ verifica cada uno    (gpt-5, cross-lab)                │
│ → strong / moderate / weak / unsupported               │
│   los claims alucinados del consenso salen en rojo     │
└────────────────────────────────────────────────────────┘
   ↓
mapa de disenso + reporte (HTML + JSON)
```

Se lee como uno de tres estados pre-registrados (definición completa de
bins en [`docs/cfi-spec.md`](docs/cfi-spec.md)):

- **aligned-stable** — los nueve agrupan estrechamente y CFI < 0.33
- **aligned-fragile** — los nueve agrupan estrechamente pero CFI ≥ 0.33
- **divided** — `σ` entre los nueve ≥ 0.03, no hay consenso real que atacar

---

## Marcos cognitivos

Nueve marcos de consenso + un disidente obligatorio. Mismo set de prompts
en cada corrida; prefijo SHA-256 expuesto como `henge.agents.PROMPTS_HASH`
y persistido en cada reporte.

| #  | Marco              | Lente                                                       | Modelo             |
|----|--------------------|-------------------------------------------------------------|--------------------|
| 1  | empirical          | cuantificación, base rates, marcadores [assumption]         | gpt-5              |
| 2  | historical         | precedentes — qué pasó las últimas 3–5 veces                | Opus 4.7           |
| 3  | first-principles   | reducir a átomos físicos / económicos / lógicos             | gpt-5              |
| 4  | analogical         | mapeos cross-domain (biología, militar, finanzas)           | Sonnet 4.6         |
| 5  | systemic           | loops de retroalimentación, efectos de 2do y 3er orden      | gpt-5              |
| 6  | ethical            | tensión deontológica + consecuencialista                    | Sonnet 4.6         |
| 7  | soft-contrarian    | reframe quirúrgico del supuesto silencioso cargado          | gpt-5              |
| 8  | radical-optimist   | qué se desbloquea si sale 10× mejor                         | gpt-5              |
| 9  | pre-mortem         | asume que falló a 12 meses — describe cómo                  | gpt-5              |
| 10a| **Décimo Hombre — ciego**     | steel-man puro · sin ver a los 9                | Opus 4.7           |
| 10b| **Décimo Hombre — informado** | ve los 9 + ciego · devuelve holds/revised/discarded | gpt-5 (cross-lab) |

El ruteo vive en `henge/config/frame_assignment.py` y es cross-lab por
diseño: la síntesis y el disenso puro se quedan en Anthropic; los roles de
auditoría cruzan a OpenAI. Sobreescribí con `FRAME_MODEL_MAP` si querés la
configuración legacy de un solo modelo.

Todos los marcos responden en el idioma de la pregunta (auto-detectado).
Forzá un locale único con `HENGE_LOCALE=en` o `HENGE_LOCALE=es`.

---

## Modelos y costos

| Etapa                          | Lab       | Modelo                       | Por qué                                                |
|--------------------------------|-----------|------------------------------|--------------------------------------------------------|
| Scoping (base)                 | Anthropic | Haiku 4.5                    | rápido, barato, ~3–5 s por llamada                     |
| Scoping (adversarial)          | OpenAI    | gpt-5                        | cross-lab — desafía supuestos ocultos                  |
| Auditoría meta-frame           | OpenAI    | gpt-5                        | clasifica pregunta; corto-circuito si exploración      |
| Contexto canónico              | Anthropic | Opus 4.7                     | resumen apretado de respuestas, marca inconsistencias  |
| 9 marcos cognitivos            | mixto     | gpt-5 ×6 / Sonnet ×2 / Opus  | razonamiento de calidad en paralelo, spread cross-lab  |
| Síntesis del consenso          | Anthropic | Haiku 4.5                    | resumen, output estructurado                           |
| Décimo hombre — ciego          | Anthropic | Opus 4.7                     | razonamiento más fuerte, sin ver a los 9               |
| Décimo hombre — informado      | OpenAI    | gpt-5                        | reconciliación cross-lab, filtro de alucinación        |
| Extracción de claims           | Anthropic | Sonnet 4.6                   | lista falsificable de claims desde el consenso         |
| Verificación de claims         | OpenAI    | gpt-5                        | califica cada claim contra los 9 marcos                |
| Embeddings                     | OpenAI    | text-embedding-3-large       | ~15–25% mejor recall en español que `-small`           |

El costo por corrida se calcula desde el `usage` real devuelto por el SDK
y se persiste en `cost_breakdown` en cada `report.json`, separado por lab
(`anthropic_usd` / `openai_usd` / `embedding_usd`) y por fase. Una corrida
v0.6 representativa cae aproximadamente en **USD 1.00–1.50**, ≈50%
Anthropic / ≈50% OpenAI. La versión de pricing se registra contra el
reporte (actualmente `2026-05`).

---

## Antes de instalar

- **Python ≥3.11.** macOS todavía trae Python 3.9. El prompt de pegado para
  Claude Code detecta esto e instala Python 3.11.9 vía pyenv automáticamente
  (sin admin/sudo, pero la compilación toma ~10 min la primera vez).
- **Dos API keys, ambas obligatorias.**
  - `ANTHROPIC_API_KEY` — Haiku 4.5, Sonnet 4.6, Opus 4.7 (3 marcos +
    décimo hombre ciego + canónico + consenso + extracción de claims).
  - `OPENAI_API_KEY` — gpt-5 (6 marcos + meta-frame + scoping adversarial
    + décimo hombre informado + verificación de claims) y
    `text-embedding-3-large`.
  v0.6 es cross-lab por diseño; ambas keys son requeridas al boot. El
  validador de arranque pinguea gpt-5 y embeddings al inicio para que
  falles ruidosamente, no 60s adentro de una llamada `/decide`.
- **Reiniciá Claude Code completamente después de instalar.** Cerrá TODAS
  las terminales corriendo `claude`, después abrilas de nuevo. El catálogo
  MCP se carga una sola vez al startup; una sesión corriendo nunca va a
  recoger un servidor recién registrado.

---

## Arquitectura cross-lab v0.6

v0.6 rutea el trabajo entre dos labs (Anthropic + OpenAI) por diseño:

- **Síntesis se queda en Anthropic** — scoping (Haiku), contexto canónico
  (Opus), consenso (Haiku), extracción de claims (Sonnet), décimo hombre
  ciego (Opus). Consistencia same-lab donde la estructura importa.
- **Auditoría cruza a OpenAI** — scoping adversarial, meta-frame, décimo
  hombre informado, verificación de claims. Cross-lab atrapa
  específicamente el caso donde el lab de síntesis aluciona: gpt-5 no
  tiene afinidad de output-style con Haiku/Sonnet y hace aflorar claims
  huérfanos.

El mapeo completo está en la tabla de Modelos y costos arriba y en
`henge/config/frame_assignment.py`.

### Migrando desde v0.5

Agregá `OPENAI_API_KEY` a `.env`. Esa es la única acción requerida.
Todo lo demás es automático — el validador de arranque pinguea gpt-5 y
embeddings al inicio, así una cuenta sin acceso falla ruidosamente al boot
en lugar de 60s adentro de una llamada `/decide`. El schema sube a `0.6`;
el lookup legacy `henge.pricing.total_cost` se mantiene para compatibilidad
hacia atrás hasta v0.7.

---

## Quickstart · Claude Code (30s)

Pegá este prompt en Claude Code y se auto-instala corriendo un script
shell determinístico — sin que el LLM siga pasos, sin drift:

````
Install Henge from https://github.com/ChrisPiz/Henge-MCP. Idempotent flow:

1. Clone shallow (or pull if already there):
   git clone --single-branch --depth 1 https://github.com/ChrisPiz/Henge-MCP.git ~/Henge \
     || (cd ~/Henge && git pull --ff-only)

2. cd ~/Henge && cp -n .env.example .env

3. Ask me for ANTHROPIC_API_KEY and OPENAI_API_KEY one at a time. When I paste each one, update the matching line in ~/Henge/.env in-place. Confirm only the LENGTH back to me ("got it, 108 chars") — never echo the value to the chat or any other tool.

4. Run the setup script — it handles Python ≥3.11 (with a 15-minute pyenv install fallback if missing), the venv, the editable install, the cross-cwd sanity check, key validation, MCP registration for every host installed (Claude Code, Claude Desktop, Cursor), and the /decide slash command:
   cd ~/Henge && ./setup

5. When the script prints "✓ Henge installed.", tell me to fully quit Claude Code (close ALL terminals running `claude`) and reopen, then try `/decide should I take the new job?`.
````

Reiniciá Claude Code completamente cuando termine, y después probá:

```
/decide ¿debería aceptar el nuevo trabajo?
```

> **Nota:** el slash command `/decide` es **solo de Claude Code**. En
> Claude Desktop y Cursor, las tools MCP no aparecen como slash commands
> — invocás Henge escribiendo tu pregunta normalmente ("¿debería renunciar
> para fundar una empresa?") y Claude levanta la tool `decide` desde su
> descripción, o podés mencionarla explícitamente ("usá la tool decide
> para analizar...").

Para Claude Desktop, Cursor o cualquier otro host MCP, ver la
[sección de Manual install en DEVELOPER.md](DEVELOPER.md#manual-install).

---

## Leyendo el reporte

Cada reporte HTML viene con dos marcadores tipo highlighter sutiles y un
toggle pill (abajo a la izquierda) para encenderlos/apagarlos:

- 🟢 **verde (conclusión)** — primer párrafo después de un encabezado
  `Conclusión / Inclinación neta / Recomendación / Veredicto / Síntesis /
  Takeaway`. _Qué creer._
- 🔵 **cyan (acción)** — bloques `<strong>` y bullets que abren con verbos
  imperativos (`Priorizar / Asignar / Empaquetar / Posponer / Resistir /
  Embeber / Decisión / Segmento / Asignación / Secuencia`, más
  equivalentes en inglés). _Qué hacer._

Heurística conservadora — pasajes sin un encabezado de conclusión explícito
o un opener imperativo se quedan sin marcar.

---

## Lo que no mide

Henge es una herramienta estructural con una métrica pre-registrada. **No**
es un instrumento validado de calidad de decisión. Específicamente: la
distancia de embedding no es desacuerdo proposicional, los diez asesores
comparten priors entre dos labs (no tres), y los thresholds del veredicto
son defaults pre-registrados, no calibrados contra ground truth.

La lista completa vive en [LIMITS.md](LIMITS.md). Leéla antes de confiar
en el output para algo load-bearing.

---

## Casos de uso

- decisiones de founders y operadores
- contratar / escalar / despedir
- estrategia y priorización de producto
- análisis de riesgo y pre-mortems
- razonamiento contrafactual
- orquestación de agentes IA donde necesitás una segunda opinión estructurada

---

## Roadmap

Trackeado en la [issues page](https://github.com/ChrisPiz/Henge-MCP/issues).

**v0.6 (actual).** Cross-lab multi-modelo — 6× gpt-5 + 2× Sonnet + 1× Opus
en los 9 marcos; scoping adversarial; contexto canónico; auditoría
meta-frame con early-exit; décimo hombre dual (ciego + informado);
verificación de claims; contabilidad de costos honesta separada por lab;
modo K-runs de distribución para CFI con `temperature > 0`; takeaway
markers en el reporte HTML.

**v0.7 (planeado).**
- Benchmark de outcomes Henge-50 (50 decisiones históricas con outcomes
  conocidos) — el reclamo de validez que v0.5 prometió y v0.6 todavía debe.
- Métricas `cross_lab_agreement` y `delta_signal` sobre el par del décimo
  hombre (high/medium/low).
- Anotación inline de claims en el cuerpo del consenso (vs. panel separado).
- Flag MCP `--force-full-run` para bypassear `reformulate` del meta-frame.
- Agregar Gemini al pool de marcos; remover el path de Voyage embeddings
  si no surge demanda.

**v0.x.** Embeddings locales (sentence-transformers, sin API key), reporte
PDF / web compartible, resultados por streaming, selección adaptiva de marcos.

---

## Fondo

Por qué existe Henge, la división rol/método, y por qué steel-man en lugar
de devil's advocate vive en [MANIFESTO.md](MANIFESTO.md).

El paper de metodología es [WHITEPAPER.md](WHITEPAPER.md).

El protocolo reproducible es [METHODOLOGY.md](METHODOLOGY.md).

---

## Referencia para developers

Para Tool API, estructura de output, instalación manual (Claude Desktop /
Cursor), config del provider de embeddings, arquitectura, y troubleshooting,
ver [DEVELOPER.md](DEVELOPER.md).

---

## Licencia

MIT
