# Henge · 9 advisors. 1 mandatory dissenter.

![Henge](docs/header-v2.jpg)

Ten pillars.
Nine align.
One must disagree.

Henge is an MCP server that measures AI consensus and forces a steel-man counter-argument before you act. Built for humans making serious decisions with AI in the loop — also drivable by autonomous agents.

**[→ See a live demo report](https://chrispiz.github.io/Henge-MCP/demo.html)**

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

## Quickstart (30s)

If you have **Claude Code** installed, paste this prompt and it self-installs:

````
Install Henge from https://github.com/ChrisPiz/Henge-MCP into ~/Henge.

Steps:
1. git clone https://github.com/ChrisPiz/Henge-MCP.git ~/Henge
2. cd into it, create a Python 3.11+ venv at .venv, activate it, pip install -r requirements.txt
3. Ask me for my ANTHROPIC_API_KEY and OPENAI_API_KEY (one at a time, don't print them back). Write them into .env using cp .env.example .env as the starting point.
4. Verify the keys by running `python -m henge.server` for ~5 seconds — it must print "✓ keys validated" to stderr. Kill it after that confirmation.
5. Register globally: `claude mcp add -s user henge "$HOME/Henge/.venv/bin/python" -- -m henge.server`
6. Confirm with `claude mcp list` — the henge row must show ✓ Connected.
7. Create the slash command at ~/.claude/commands/decide.md with this content:
   ---
   description: Invokes Henge — disagreement map of 9 advisors + 1 dissenter.
   ---
   Use the `decide` MCP tool from the `henge` server to analyze: $ARGUMENTS. When the JSON returns: cite viz_path, summarize the consensus first, list the 9 advisors' conclusions, then quote the tenth-man verbatim.

After step 6, tell me to restart Claude Code and try `/decide should I take the new job?`
````

Restart Claude Code once when it's done, then try:

```
/decide should I take the new job?
```

For Claude Desktop, Cursor, or a manual setup, see the [Install matrix](#install-matrix) below.

---

## Install matrix

| Client          | Install                                          |
| --------------- | ------------------------------------------------ |
| Claude Code     | One-shot — see [Quickstart](#quickstart-30s)     |
| Claude Desktop  | Manual config edit                               |
| Cursor          | Manual config edit                               |
| Anything else   | Manual `pip install -e .` + run as MCP server    |

All paths reach the same MCP server and call the same `decide` tool. Reports persist at `~/.henge/reports/` and the browseable `index.html` ledger auto-regenerates on every run.

### Claude Desktop · manual

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "henge": {
      "command": "python",
      "args": ["-m", "henge.server"],
      "cwd": "/absolute/path/to/Henge-MCP",
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

### Cursor · manual

Add the same `mcpServers.henge` block to Cursor's MCP config (`Settings → MCP → Edit`).

### Manual install (any environment)

```bash
git clone https://github.com/ChrisPiz/Henge-MCP.git
cd Henge-MCP
pip install -e .

cp .env.example .env
# edit .env:
#   ANTHROPIC_API_KEY  (required — runs the 9 frames + tenth-man)
#   OPENAI_API_KEY     (default embedding provider)
#   VOYAGE_API_KEY     (optional — set EMBED_PROVIDER=voyage for higher quality)

python -m henge.server   # runs as an MCP stdio server
```

---

## MCP integration

Henge speaks Model Context Protocol. Any MCP-compatible client can drive it as a reasoning tool.

Tested with:

- **Claude Code** (one-shot install)
- **Claude Desktop** (manual config)
- **Cursor** (manual config)
- Any other MCP-compatible agent or local AI pipeline

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
│ embeddings (OpenAI / Voyage)  │
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
| Embeddings         | OpenAI / Voyage      | `text-embedding-3-small` by default|

Typical cost per full run: **~USD 0.65** (range USD 0.50–0.80 depending on token spread).

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
