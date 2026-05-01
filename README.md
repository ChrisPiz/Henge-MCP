# Henge · 9 advisors. 1 mandatory dissenter.

![Henge](docs/header-v2.jpg)

Ten pillars.
Nine align.
One must disagree.

Henge is an MCP server that measures AI consensus and forces a steel-man counter-argument before you act. Built for humans making serious decisions with AI in the loop — also drivable by autonomous agents.

**[→ See a live demo report](https://chrispiz.github.io/Henge/demo.html)**

---

## Quickstart · Claude Code (30s)

Paste this prompt into Claude Code and it self-installs by running a deterministic shell script — no LLM step-following, no drift:

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

For Claude Desktop, Cursor or any other MCP host, see [Manual install](#manual-install) at the bottom.

---

## The problem

AI systems don't fail because they lack intelligence.

They fail because:
- they converge too fast
- they reinforce assumptions
- they mistake agreement for truth

Consensus is cheap. Correct decisions are not.

---

## What Henge does

Henge runs your question through ten cognitive perspectives and:

1. Asks 4–7 scoping questions before reasoning, so the advisors apply to facts instead of speculation
2. Runs nine cognitive frames in parallel — each with its own lens
3. Embeds the answers, projects them with classical MDS, and measures cosine distance to the centroid of the nine
4. Forces a tenth advisor to steel-man the dissent against whatever consensus emerged
5. Persists a full HTML report + JSON record on disk and opens it in your browser

---

## Why this is different

| Approach              | Problem                       | Henge                              |
| --------------------- | ----------------------------- | ---------------------------------- |
| Single LLM            | Overconfident answers         | Multi-frame reasoning              |
| Multi-agent debate    | Noisy, redundant              | Measures structure, doesn't echo   |
| Devil's advocate      | Always contradicts            | Tenth-man only when warranted      |
| Fixed "tenth man" rule| Hard-coded contrarian         | Steel-man with measurable distance |

---

## Core principle

Forcing disagreement without consensus is noise.

Henge does not simulate debate. It analyzes the structure of thought, then quantifies the distance between voices so the dissent has somewhere to land.

---

## Before you install

Quick checklist so the install doesn't surprise you:

- **Python ≥3.11.** macOS still ships Python 3.9. The Claude Code paste prompt detects this and installs Python 3.11.9 via pyenv automatically (no admin/sudo, but the build takes ~10 min the first time).
- **Two API keys.** `ANTHROPIC_API_KEY` (mandatory — runs the 10 advisors) and `OPENAI_API_KEY` (embedding provider).
- **Restart Claude Code fully after install.** Close ALL terminals running `claude`, then reopen. The MCP catalog is loaded once at startup; a running session will never pick up a freshly-registered server.

---

## How it works

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
│ tenth-man steel-man (Opus)    │
│ ↓                             │
│ disagreement map + report     │
└───────────────────────────────┘
```

The verdict is one of three states:

- **aligned-stable** — the nine cluster tightly and the tenth's dissent is moderate
- **aligned-fragile** — the nine are tight but the tenth pushes far enough to break it coherently
- **divided** — the nine themselves are spread; there was no real consensus to attack

---

## Cognitive frames

Nine consensus frames + one mandatory dissenter:

| # | Frame              | Lens                                                      |
|---|--------------------|-----------------------------------------------------------|
| 1 | empirical          | quantification, base rates, [assumption] markers          |
| 2 | historical         | precedents — what happened the last 3–5 times             |
| 3 | first-principles   | reduce to physical / economic / logical atoms             |
| 4 | analogical         | cross-domain mappings (biology, military, finance)        |
| 5 | systemic           | feedback loops, second- and third-order effects           |
| 6 | ethical            | deontological + consequentialist tension                  |
| 7 | soft-contrarian    | surgical reframe of the loaded silent assumption          |
| 8 | radical-optimist   | what unlocks if it goes 10× better                        |
| 9 | pre-mortem         | assume it failed in 12 months — describe how              |
| 10| **tenth-man**      | steel-man dissent, mandatory, after the nine align        |

All frames respond in the **same language as the question** (Spanish question → Spanish answer; English → English). The report chrome (headings, labels, reading guide) follows the same rule by auto-detecting the question's language; force a single locale with `HENGE_LOCALE=en` or `HENGE_LOCALE=es` in your `.env`.

---

## Models & costs

| Stage              | Model                | Why                                |
| ------------------ | -------------------- | ---------------------------------- |
| Scoping            | Claude Haiku 4.5     | fast, cheap, ~3–5 s per call       |
| 9 cognitive frames | Claude Sonnet 4.6    | quality reasoning, parallel        |
| Consensus synthesis| Claude Haiku 4.5     | summarization, structured output   |
| Tenth-man dissent  | Claude Opus 4.7      | hardest reasoning, fully sequential|
| Embeddings         | OpenAI               | `text-embedding-3-small` by default|

Typical cost per full run: **~USD 0.65** (range USD 0.50–0.80 depending on token spread).

---

## Use cases

- founder & operator decisions
- hiring / scaling / firing
- product strategy and prioritization
- risk analysis & pre-mortems
- counterfactual reasoning
- AI agent orchestration where you need a structured second opinion

---

## What this is NOT

- not a chatbot
- not a debate simulator
- not a multi-agent chat
- not a vibe-checker

It is a **decision-quality** tool. The output is a measurable structure of agreement and disagreement, not a longer answer.

---

## Roadmap

- numeric consensus-strength scoring
- dissent-impact scoring
- adaptive frame selection (only run the lenses that matter)
- PDF / shareable web report
- streaming results
- multi-model support (Gemini, GPT, local)
- local embeddings (sentence-transformers, no API key required)

---

## Design philosophy

- don't generate more answers → generate better structure
- don't simulate intelligence → measure it
- don't force dissent → earn it

---

## Mental model

Henge is not trying to be right.

It is trying to make your thinking harder to break.

---

## License

MIT

---
---

# Developer reference

Everything below is for integrators and developers. The [Quickstart](#quickstart--claude-code-30s) at the top is enough for normal use.

## Manual install

| Client          | Install                                          |
| --------------- | ------------------------------------------------ |
| Claude Code     | One-shot — see [Quickstart](#quickstart--claude-code-30s) |
| Claude Desktop  | Manual config edit                               |
| Cursor          | Manual config edit                               |
| Anything else   | Manual `pip install -e .` + run as MCP server    |

All paths reach the same MCP server and call the same `decide` tool. Reports persist at `~/.henge/reports/` and the browseable `index.html` ledger auto-regenerates on every run.

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "henge": {
      "command": "python",
      "args": ["-m", "henge.server"],
      "cwd": "/absolute/path/to/Henge",
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

### Cursor

Add the same `mcpServers.henge` block to Cursor's MCP config (`Settings → MCP → Edit`).

### Any environment

```bash
git clone --single-branch --depth 1 https://github.com/ChrisPiz/Henge.git ~/Henge
cd ~/Henge
cp .env.example .env
# Open .env and fill in:
#   ANTHROPIC_API_KEY  (required — runs the 9 frames + tenth-man)
#   OPENAI_API_KEY     (embedding provider)

./setup                  # does Python detection, venv, pip install -e .,
                         # key validation, MCP registration, slash command
```

The `setup` script handles every step deterministically. If you'd rather run the pieces yourself: `python3.11 -m venv .venv && .venv/bin/pip install -e .` then register the MCP with your host of choice.

### Setup script flags (optional)

The Quickstart paste runs `./setup` with no flags — the script auto-detects every MCP host on the machine (Claude Code, Claude Desktop, Cursor) and registers Henge for each. Use these flags only to override that default.

```bash
./setup                          # default — auto-detect installed hosts (what the Quickstart runs)
./setup --host claude-code       # register only for Claude Code
./setup --host claude-desktop    # register only for Claude Desktop
./setup --host cursor            # register only for Cursor
./setup --host all               # register for every host even if not detected
./setup --skip-validate          # skip the API-key validation step
```

---

## Tool API

### `decide(question, context=None, skip_scoping=False)`

Two-phase. Phase 1 returns clarifying questions; phase 2 runs the ten advisors with the answers as context.

#### Phase 1 — scoping (default)

```jsonc
// call
{ "question": "Should I hire someone now?" }

// response
{
  "status": "needs_context",
  "questions": [
    "What is your approximate net monthly income?",
    "What's your runway in months?",
    "Is the role revenue-generating or cost-saving?",
    "..."
  ],
  "next_call_hint": "decide(question='...', context='<user answers formatted>')"
}
```

#### Phase 2 — run

```jsonc
// call
{
  "question": "Should I hire someone now?",
  "context": "Net income USD 2.6K / month, runway 8 months, role: senior engineer, ..."
}

// response: full disagreement-map JSON (see `Output structure` below)
```

#### Skip scoping

```jsonc
{ "question": "...", "skip_scoping": true }   // when the question already has rich context
```

---

## Example usage (agent)

```ts
const phase1 = await mcp.tools.decide({
  question: "Should I expand my business?"
})
// phase1.questions → present to user, collect answers

const phase2 = await mcp.tools.decide({
  question: "Should I expand my business?",
  context: "Revenue USD 2.6K, expenses 550, 8 months runway, ..."
})
// phase2.viz_path → opens HTML
// phase2.summary.consensus_state → drives downstream agent logic
```

---

## Output structure

```jsonc
{
  "viz_path": "/Users/you/.henge/reports/20260501-2247_should-i-hire-now/report.html",
  "report_id": "20260501-2247_should-i-hire-now",
  "report_dir": "/Users/you/.henge/reports/20260501-2247_should-i-hire-now",
  "consensus": "# Validate before hiring — asymmetric risk dominates\n\n## (1) Where the nine converge ...",
  "frames": [
    { "frame": "empirical",        "status": "ok", "distance": 0.046, "summary": "..." },
    { "frame": "first-principles", "status": "ok", "distance": 0.069, "summary": "..." }
    // 7 more
  ],
  "tenth_man": {
    "distance": 0.148,
    "response": "## §1 Facts I accept\n... ## §2 Where the consensus fails ..."
  },
  "summary": {
    "tenth_man_distance": 0.148,
    "max_frame_distance": 0.085,
    "consensus_state": "aligned-stable",       // or "aligned-fragile" | "divided"
    "consensus_fragility": "Advisors aligned — dissent sounds reasonable but consensus holds.",
    "n_frames_succeeded": 9,
    "embed_provider": "openai",
    "embed_model": "text-embedding-3-small"
  },
  "cost_usd": 0.65
}
```

The HTML at `viz_path` ships with the disagreement map, sortable frames table, consensus card, tenth-man steel-man, and a per-run hero painting bundled inside `report_dir/assets/`.

---

## Reports & ledger

Each run writes:

```
~/.henge/reports/
  20260501-224712_should-i-hire-now/
    report.html       # full editorial visualization
    report.json       # canonical record (question, context, 10 responses, distances, summary)
    assets/
      header-v2.jpg   # bundled hero painting
  index.html          # auto-regenerated ledger of every past report
```

The JSON is the source of truth. The HTML is a pure render of it — delete a directory and it disappears from the ledger on the next run.

---

## Architecture

```
henge/
  agents.py        # 9 frames in parallel + tenth-man sequencing
  embed.py         # provider-agnostic embeddings + classical MDS
  scoping.py       # 4–7 clarifying questions (Haiku)
  consensus.py     # synthesis of the nine (Haiku)
  viz.py           # editorial HTML report + disagreement map SVG
  storage.py       # report.json + report.html + ledger
  server.py        # MCP entrypoint
  prompts/         # 10 cognitive-frame markdowns
  assets/          # bundled hero painting
```

---

## Embeddings · provider config

Henge needs to compute distance between the 10 advisor responses to draw the disagreement map, and that requires **embeddings** (text → vector). Anthropic does not currently offer an embeddings API, so a second provider is unavoidable for the math layer.

**Default: OpenAI `text-embedding-3-small`** — ~USD 0.0005/run. Most devs already have a key. Recommended.

**Voyage AI (advanced opt-in).** Higher embedding quality for Spanish, but the free tier is unusable: without a payment method on file, Voyage caps at **3 RPM / 10K TPM**, and a single Henge run embeds 10 advisor responses in one call — that trips the TPM cap and fails the run. To use Voyage you must add a payment method on https://dashboard.voyageai.com/ first (the 200M free tokens/month still apply, the payment method only unlocks standard rate limits). Then set `EMBED_PROVIDER=voyage` + `VOYAGE_API_KEY=...` in `.env`. Otherwise leave it on OpenAI.

A local-embeddings option (no API key, sentence-transformers on-device) is on the [Roadmap](#roadmap).

---

## MCP integration

Henge speaks Model Context Protocol. Any MCP-compatible client can drive it as a reasoning tool.

Tested with:

- **Claude Code** (one-shot install)
- **Claude Desktop** (manual config)
- **Cursor** (manual config)
- Any other MCP-compatible agent or local AI pipeline

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `/decide` says "Henge MCP server doesn't appear to be connected" | Quit every `claude` session and reopen. If still failing → next row. |
| `claude mcp list` shows `henge ✗ Failed` | Run `~/Henge/.venv/bin/python -m henge.server </dev/null` interactively to see the real error. Most common cause: the editable install was skipped — `cd ~/Henge && .venv/bin/pip install -e .` and re-register. |
| `ModuleNotFoundError: No module named 'henge'` | Editable install missing. `cd ~/Henge && .venv/bin/pip install -e .` |
| `git clone` fails with "destination already exists" | The repo is already there. `cd ~/Henge && git pull --ff-only`. |
| `pyenv install` shows an `lzma` warning | Non-fatal, ignore. (Optional cleanup: `brew install xz` and re-run.) |
| Server validates keys then exits immediately | Expected. It's a stdio MCP server; without a client on stdin, it shuts down once `✓ keys validated` is printed. |
| `claude mcp add` rejects the path with $HOME | Pass an absolute path: `~/Henge/.venv/bin/python` (Claude Code expands `~`, but other shells re-expand `$HOME` after registration). |
| `embed_failed` with Voyage `RateLimitError` (3 RPM / 10K TPM) | Voyage free tier without payment method can't run Henge. Add a payment method at https://dashboard.voyageai.com/ — or remove `EMBED_PROVIDER=voyage` from `.env` to fall back to OpenAI. |
