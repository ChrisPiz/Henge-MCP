# Henge

**Structured Dissent Protocol** — 9 cognitive frames + 1 mandatory steel-man
dissenter, exposed as an MCP server.

![Henge](docs/header-v2.jpg)

Henge runs your decision through nine cognitive lenses in parallel, then
forces a tenth advisor — the **Tenth Man** — to build the strongest possible
steel-man case against whatever consensus the nine reach. It persists a JSON
record + an editorial HTML report on disk and surfaces a single tri-state
verdict driven by the **Consensus Fragility Index (CFI)**.

**Claim (v0.5.0).** Given a pre-registered protocol with `temperature=0`,
the same question produces the same CFI bin across runs, modulo numerical
noise from MDS init. Validation against decision quality (Henge-50 benchmark)
is planned for v0.6.

[Demo](https://chrispiz.github.io/Henge/demo.html) · [Paper](WHITEPAPER.md) ·
[Limits](LIMITS.md) · [Methodology](METHODOLOGY.md) · [CFI spec](docs/cfi-spec.md) ·
[Manifesto](MANIFESTO.md) · [Developer](DEVELOPER.md)

---

## What it does

```
question
   ↓
┌─ phase 1 ─────────────────────┐
│ scoping (Haiku 4.5)           │
│ → 4–7 clarifying questions    │
└───────────────────────────────┘
   ↓ user answers
┌─ phase 2 ─────────────────────┐
│ 9 frames in parallel (Sonnet) │
│ ↓                             │
│ embeddings (OpenAI)           │
│ ↓                             │
│ classical MDS + cosine        │
│ ↓                             │
│ consensus synthesis (Haiku)   │
│ ↓                             │
│ Tenth Man via steel-man (Opus)│
│ ↓                             │
│ disagreement map + report     │
└───────────────────────────────┘
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

| #  | Frame              | Lens                                                      |
|----|--------------------|-----------------------------------------------------------|
| 1  | empirical          | quantification, base rates, [assumption] markers          |
| 2  | historical         | precedents — what happened the last 3–5 times             |
| 3  | first-principles   | reduce to physical / economic / logical atoms             |
| 4  | analogical         | cross-domain mappings (biology, military, finance)        |
| 5  | systemic           | feedback loops, second- and third-order effects           |
| 6  | ethical            | deontological + consequentialist tension                  |
| 7  | soft-contrarian    | surgical reframe of the loaded silent assumption          |
| 8  | radical-optimist   | what unlocks if it goes 10× better                        |
| 9  | pre-mortem         | assume it failed in 12 months — describe how              |
| 10 | **Tenth Man**      | mandatory dissent role · method: steel-man, after the nine align |

All frames respond in the language of the question (auto-detected). Force a
single locale with `HENGE_LOCALE=en` or `HENGE_LOCALE=es`.

---

## Models & costs

| Stage              | Model                | Why                                |
|--------------------|----------------------|------------------------------------|
| Scoping            | Claude Haiku 4.5     | fast, cheap, ~3–5 s per call       |
| 9 cognitive frames | Claude Sonnet 4.6    | quality reasoning, parallel        |
| Consensus synthesis| Claude Haiku 4.5     | summarisation, structured output   |
| Tenth-man dissent  | Claude Opus 4.7      | hardest reasoning, fully sequential|
| Embeddings         | OpenAI               | `text-embedding-3-small` by default|

Cost per run is computed from real `usage` returned by the SDK and persisted
under `cost_breakdown` in every `report.json`. A representative full run with
all 10 advisors landing inside their token caps lands at roughly **USD 0.50–0.80**.
Pricing version is recorded against the report (currently `2026-05-01`).

---

## Before you install

- **Python ≥3.11.** macOS still ships Python 3.9. The Claude Code paste prompt
  detects this and installs Python 3.11.9 via pyenv automatically (no admin/sudo,
  but the build takes ~10 min the first time).
- **Two API keys.** `ANTHROPIC_API_KEY` (mandatory — runs the 10 advisors)
  and `OPENAI_API_KEY` (embedding provider).
- **Restart Claude Code fully after install.** Close ALL terminals running
  `claude`, then reopen. The MCP catalog is loaded once at startup; a running
  session will never pick up a freshly-registered server.

---

## v0.6 cross-lab architecture

v0.6 routes the 9 cognitive frames across two model families (Anthropic + OpenAI)
and adds three cross-lab audit layers (adversarial scoping, meta-frame,
tenth-man informed, claim verification). The split is intentional: synthesis
tasks stay in Anthropic (consensus, canonical context, blind dissent); audit
tasks cross to OpenAI (challenge the question, verify claims, refine the
dissent). When the synthesizer hallucinates, the cross-lab auditor catches it.

**Cost per `/decide` run:** USD ~1.00–1.50 (≈50% Anthropic / ≈50% OpenAI).

**Required keys:**
- `ANTHROPIC_API_KEY` (Haiku 4.5, Sonnet 4.6, Opus 4.7)
- `OPENAI_API_KEY` (gpt-5 + text-embedding-3-large)

### Migrating from v0.5

Add `OPENAI_API_KEY` to `.env`. That's it. Everything else is automatic.
The startup validator now pings gpt-5 and embeddings to confirm access; if
your account doesn't have gpt-5, you'll see a clear error at boot instead
of a 60s mystery failure during invocation.

---

## Quickstart · Claude Code (30s)

Paste this prompt into Claude Code and it self-installs by running a deterministic
shell script — no LLM step-following, no drift:

````
Install Henge from https://github.com/ChrisPiz/Henge. Idempotent flow:

1. Clone shallow (or pull if already there):
   git clone --single-branch --depth 1 https://github.com/ChrisPiz/Henge.git ~/Henge \
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

Tracked in the [issues page](https://github.com/ChrisPiz/Henge/issues). Major
items beyond v0.5:

- **v0.6** · Henge-50 outcome benchmark (50 historical decisions with known
  outcomes), embedding token accounting, K-runs distribution mode for CFI
  with `temperature > 0`.
- **v0.7** · Multi-model real (Gemini / GPT in the frame pool).
- **v0.x** · local embeddings (sentence-transformers, no API key), PDF /
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
