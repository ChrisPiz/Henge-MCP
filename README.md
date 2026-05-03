# Henge

> **Read this in:** [English](README.md) · [Español](README.es.md)

**Structured Dissent Protocol** — 9 cognitive frames + 1 mandatory steel-man
dissenter, exposed as an MCP server.

![Henge](docs/header-v2.jpg)

Henge runs your decision through nine cognitive lenses in parallel, then
forces a tenth advisor — the **Tenth Man** — to build the strongest possible
steel-man case against whatever consensus the nine reach. It persists a JSON
record + an editorial HTML report on disk and surfaces a single tri-state
verdict driven by the **Consensus Fragility Index (CFI)**.

**Claim (v0.6.0).** Given a pre-registered protocol with `temperature=0`,
the same question produces the same CFI bin across runs, modulo numerical
noise from MDS init. v0.6 splits the run across two labs (Anthropic +
OpenAI) by design, so the synthesizer's hallucinations get caught by the
auditor instead of laundered into consensus. Validation against decision
quality (Henge-50 benchmark) remains in flight.

[Demo](https://chrispiz.github.io/Henge-MCP/demo.html) · [Paper](WHITEPAPER.md) ·
[Limits](LIMITS.md) · [Methodology](METHODOLOGY.md) · [CFI spec](docs/cfi-spec.md) ·
[Manifesto](MANIFESTO.md) · [Developer](DEVELOPER.md)

---

## What it does

```
question
   ↓
┌─ phase 1 · scoping ────────────────────────────────────┐
│ base questions      (Haiku 4.5)                        │
│ adversarial sweep   (gpt-5, cross-lab)                 │
│ → 4–7 questions, 2–4 of them challenging hidden        │
│   assumptions in the question itself                   │
└────────────────────────────────────────────────────────┘
   ↓ user answers
┌─ phase 2 · meta-frame ─────────────────────────────────┐
│ classify (gpt-5, cross-lab)                            │
│ → decision_class · urgency · question_quality          │
│   · meta_recommendation                                │
│ if proxy / exploration / fake-urgency:                 │
│   short-circuit with suggested_reformulation           │
│   (saves ~$1.00/run)                                   │
└────────────────────────────────────────────────────────┘
   ↓
┌─ phase 3 · canonical context ──────────────────────────┐
│ canonicalize answers   (Opus 4.7)                      │
│ → tight executive summary + flagged inconsistencies    │
│   shown to all 9 advisors                              │
└────────────────────────────────────────────────────────┘
   ↓
┌─ phase 4 · 9 frames in parallel ───────────────────────┐
│ 6× gpt-5 + 2× Sonnet 4.6 + 1× Opus 4.7                 │
│ ↓                                                      │
│ embeddings (text-embedding-3-large)                    │
│ ↓                                                      │
│ classical MDS + cosine                                 │
└────────────────────────────────────────────────────────┘
   ↓
┌─ phase 5 · synthesis + dual dissent ───────────────────┐
│ consensus           (Haiku 4.5)                        │
│ tenth-man blind     (Opus 4.7, no view of the 9)       │
│ tenth-man informed  (gpt-5, cross-lab — sees the 9     │
│                      + blind, returns what_holds /     │
│                      what_revised / what_discarded)    │
└────────────────────────────────────────────────────────┘
   ↓
┌─ phase 6 · claim verification ─────────────────────────┐
│ extract claims      (Sonnet 4.6)                       │
│ verify each         (gpt-5, cross-lab)                 │
│ → strong / moderate / weak / unsupported               │
│   hallucinated consensus claims surface in red         │
└────────────────────────────────────────────────────────┘
   ↓
disagreement map + report (HTML + JSON)
```

Reads as one of three pre-registered states (full bin definition in
[`docs/cfi-spec.md`](docs/cfi-spec.md)):

- **aligned-stable** — the nine cluster tightly and CFI < 0.33
- **aligned-fragile** — the nine cluster tightly but CFI ≥ 0.33
- **divided** — `σ` across the nine ≥ 0.03, no real consensus to attack

---

## Cognitive frames

Nine consensus frames + one mandatory dissenter. Same prompt set across every
run; SHA-256 prefix exposed as `henge.agents.PROMPTS_HASH` and persisted in
every report.

| #  | Frame              | Lens                                                      | Model              |
|----|--------------------|-----------------------------------------------------------|--------------------|
| 1  | empirical          | quantification, base rates, [assumption] markers          | gpt-5              |
| 2  | historical         | precedents — what happened the last 3–5 times             | Opus 4.7           |
| 3  | first-principles   | reduce to physical / economic / logical atoms             | gpt-5              |
| 4  | analogical         | cross-domain mappings (biology, military, finance)        | Sonnet 4.6         |
| 5  | systemic           | feedback loops, second- and third-order effects           | gpt-5              |
| 6  | ethical            | deontological + consequentialist tension                  | Sonnet 4.6         |
| 7  | soft-contrarian    | surgical reframe of the loaded silent assumption          | gpt-5              |
| 8  | radical-optimist   | what unlocks if it goes 10× better                        | gpt-5              |
| 9  | pre-mortem         | assume it failed in 12 months — describe how              | gpt-5              |
| 10a| **Tenth Man — blind**    | pure steel-man · no view of the 9                | Opus 4.7           |
| 10b| **Tenth Man — informed** | sees 9 + blind · returns holds/revised/discarded | gpt-5 (cross-lab)  |

Routing lives in `henge/config/frame_assignment.py` and is cross-lab by
design: synthesis and pure dissent stay in Anthropic; audit roles cross to
OpenAI. Override via `FRAME_MODEL_MAP` if you want the legacy single-model
configuration.

All frames respond in the language of the question (auto-detected). Force a
single locale with `HENGE_LOCALE=en` or `HENGE_LOCALE=es`.

---

## Models & costs

| Stage                       | Lab       | Model                       | Why                                                    |
|-----------------------------|-----------|-----------------------------|--------------------------------------------------------|
| Scoping (base)              | Anthropic | Haiku 4.5                   | fast, cheap, ~3–5 s per call                           |
| Scoping (adversarial)       | OpenAI    | gpt-5                       | cross-lab — challenges hidden assumptions              |
| Meta-frame audit            | OpenAI    | gpt-5                       | classify question; short-circuit if exploration/proxy  |
| Canonical context           | Anthropic | Opus 4.7                    | tight summary of user answers, flag inconsistencies    |
| 9 cognitive frames          | mixed     | gpt-5 ×6 / Sonnet ×2 / Opus | quality reasoning in parallel, cross-lab spread        |
| Consensus synthesis         | Anthropic | Haiku 4.5                   | summarisation, structured output                       |
| Tenth-man — blind           | Anthropic | Opus 4.7                    | hardest reasoning, no view of the 9                    |
| Tenth-man — informed        | OpenAI    | gpt-5                       | cross-lab reconciliation, hallucination filter         |
| Claim extraction            | Anthropic | Sonnet 4.6                  | falsifiable claim list from consensus                  |
| Claim verification          | OpenAI    | gpt-5                       | rate each claim against the 9 frame outputs            |
| Embeddings                  | OpenAI    | text-embedding-3-large      | ~15–25% better Spanish recall than `-small`            |

Cost per run is computed from real `usage` returned by the SDK and persisted
under `cost_breakdown` in every `report.json`, split by lab
(`anthropic_usd` / `openai_usd` / `embedding_usd`) and by phase. A
representative full v0.6 run lands at roughly **USD 1.00–1.50**, ≈50%
Anthropic / ≈50% OpenAI. Pricing version is recorded against the report
(currently `2026-05`).

---

## Before you install

- **Python ≥3.11.** macOS still ships Python 3.9. The Claude Code paste prompt
  detects this and installs Python 3.11.9 via pyenv automatically (no admin/sudo,
  but the build takes ~10 min the first time).
- **Two API keys, both mandatory.**
  - `ANTHROPIC_API_KEY` — Haiku 4.5, Sonnet 4.6, Opus 4.7 (3 frames + blind
    tenth-man + canonical + consensus + claims-extract).
  - `OPENAI_API_KEY` — gpt-5 (6 frames + meta-frame + adversarial scoping +
    informed tenth-man + claim verification) and `text-embedding-3-large`.
  v0.6 is cross-lab by design; both keys are required at boot. The
  startup validator pings gpt-5 and embeddings up front so you fail
  loudly, not 60s into a `/decide` call.
- **Restart Claude Code fully after install.** Close ALL terminals running
  `claude`, then reopen. The MCP catalog is loaded once at startup; a running
  session will never pick up a freshly-registered server.

---

## v0.6 cross-lab architecture

v0.6 routes work across two labs (Anthropic + OpenAI) by design:

- **Synthesis stays in Anthropic** — scoping (Haiku), canonical context
  (Opus), consensus (Haiku), claims extraction (Sonnet), tenth-man blind
  (Opus). Same-lab consistency where structure matters.
- **Audit crosses to OpenAI** — adversarial scoping, meta-frame, tenth-man
  informed, claim verification. Cross-lab specifically catches the case
  where the synth lab hallucinates: gpt-5 has no output-style affinity
  with Haiku/Sonnet and surfaces orphan claims.

The full mapping is in the Models & costs table above and in
`henge/config/frame_assignment.py`.

### Migrating from v0.5

Add `OPENAI_API_KEY` to `.env`. That's the only required action.
Everything else is automatic — the startup validator pings gpt-5 and
embeddings up front, so a missing-access account fails loudly at boot
instead of 60s into a `/decide` call. Schema bumps to `0.6`; the legacy
`henge.pricing.total_cost` lookup is kept for back-compat through v0.7.

---

## Quickstart · Claude Code (30s)

Paste this prompt into Claude Code and it self-installs by running a deterministic
shell script — no LLM step-following, no drift:

````
Install Henge from https://github.com/ChrisPiz/Henge. Idempotent flow:

1. Clone shallow (or pull if already there):
   git clone --single-branch --depth 1 https://github.com/ChrisPiz/Henge-MCP.git ~/Henge \
     || (cd ~/Henge && git pull --ff-only)

2. cd ~/Henge && cp -n .env.example .env

3. Ask me for ANTHROPIC_API_KEY and OPENAI_API_KEY one at a time. When I paste each one, update the matching line in ~/Henge/.env in-place. Confirm only the LENGTH back to me ("got it, 108 chars") — never echo the value to the chat or any other tool.

4. Run the setup script — it handles Python ≥3.11 (with a 15-minute pyenv install fallback if missing), the venv, the editable install, the cross-cwd sanity check, key validation, MCP registration for every host installed (Claude Code, Claude Desktop, Cursor), and the /decide slash command:
   cd ~/Henge && ./setup

5. When the script prints "✓ Henge installed.", tell me to fully quit Claude Code (close ALL terminals running `claude`) and reopen, then try `/decide should I take the new job?`.
````

Restart Claude Code fully when it's done, then try:

```
/decide should I take the new job?
```

> **Note:** the `/decide` slash command is **Claude Code only**. In Claude
> Desktop and Cursor, MCP tools don't appear as slash commands — you invoke
> Henge by writing your question normally ("Should I quit my job to start a
> company?") and Claude picks up the `decide` tool from its description, or
> you can mention it explicitly ("use the decide tool to analyze ...").

For Claude Desktop, Cursor or any other MCP host, see the
[Manual install section in DEVELOPER.md](DEVELOPER.md#manual-install).

---

## Reading the report

Every HTML report ships with two subtle highlighter-style markers and a
toggle pill (bottom-left) to switch them on/off:

- 🟢 **green (conclusion)** — first paragraph after a `Conclusión /
  Inclinación neta / Recomendación / Veredicto / Síntesis / Takeaway`
  heading. _What to believe._
- 🔵 **cyan (action)** — `<strong>` blocks and bullets opening with
  imperative verbs (`Priorizar / Asignar / Empaquetar / Posponer / Resistir /
  Embeber / Decisión / Segmento / Asignación / Secuencia`, plus English
  equivalents). _What to do._

Conservative heuristic — passages without an explicit conclusion heading
or imperative opener stay unmarked.

---

## What it does not measure

Henge is a structural tool with a pre-registered metric. It is **not** a
validated decision-quality instrument. Specifically: embedding distance is
not propositional disagreement, all ten advisors share an Anthropic backbone,
and verdict thresholds are pre-registered defaults rather than calibrated
against ground truth.

The full list lives in [LIMITS.md](LIMITS.md). Read it before you trust the
output for anything load-bearing.

---

## Use cases

- founder & operator decisions
- hiring / scaling / firing
- product strategy and prioritisation
- risk analysis & pre-mortems
- counterfactual reasoning
- AI agent orchestration where you need a structured second opinion

---

## Roadmap

Tracked in the [issues page](https://github.com/ChrisPiz/Henge-MCP/issues).

**v0.6 (current).** Cross-lab multi-model — 6× gpt-5 + 2× Sonnet + 1× Opus
across the 9 frames; adversarial scoping; canonical context; meta-frame
audit with early-exit; dual tenth-man (blind + informed); claim
verification; honest cost accounting split by lab; K-runs distribution
mode for CFI with `temperature > 0`; takeaway markers in the HTML report.

**v0.7 (planned).**
- Henge-50 outcome benchmark (50 historical decisions with known outcomes)
  — the validity claim that v0.5 promised and v0.6 still owes.
- `cross_lab_agreement` and `delta_signal` metrics on the tenth-man pair
  (high/medium/low).
- Inline claim annotation in the consensus body (vs. a separate panel).
- `--force-full-run` MCP flag to bypass meta-frame `reformulate`.
- Add Gemini to the frame pool; remove the Voyage embeddings path if
  no demand surfaces.

**v0.x.** Local embeddings (sentence-transformers, no API key), PDF /
shareable web report, streaming results, adaptive frame selection.

---

## Background

Why Henge exists, the role/method split, and why steel-man over devil's
advocate live in [MANIFESTO.md](MANIFESTO.md).

The methodology paper is [WHITEPAPER.md](WHITEPAPER.md).

The reproducible protocol is [METHODOLOGY.md](METHODOLOGY.md).

---

## Developer reference

For Tool API, output structure, manual installs (Claude Desktop / Cursor),
embeddings provider config, architecture, and troubleshooting, see
[DEVELOPER.md](DEVELOPER.md).

---

## License

MIT
