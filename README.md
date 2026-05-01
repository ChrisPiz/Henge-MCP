# Henge · Structured Dissent for AI Agents

Henge is a Model Context Protocol server that helps AI agents avoid premature consensus.

It runs multiple cognitive frames over a decision, measures alignment, and generates structured dissent only when needed.

> Agreement is not a signal. It's just coherent noise — unless you measure it.

**[→ Live demo report](https://raw.githack.com/ChrisPiz/Henge/main/docs/demo.html)** — see what Henge returns for a real founder decision.

---

## What it does

Henge gives compatible AI clients a tool for decision stress-testing.

Instead of asking one model for one answer, it:

1. Generates multiple cognitive perspectives
2. Measures agreement between them
3. Detects whether consensus is strong, weak, or fragmented
4. Produces a steel-man dissent when useful
5. Returns a structured decision report

---

## Why MCP

MCP makes Henge available as a reusable reasoning tool inside:

- Claude Desktop
- Cursor
- local AI workflows
- agent builders
- custom orchestration systems

The goal is not to replace the main AI assistant.

The goal is to give it a specialized tool for:

- avoiding groupthink
- questioning consensus
- comparing assumptions
- improving decision quality

---

## Setup

### Option A · One-shot install with Claude Code (recommended)

Paste this prompt into Claude Code and let it do the install for you:

```
Install Henge from https://github.com/ChrisPiz/Henge into ~/Henge.

Steps:
1. git clone the repo into ~/Henge
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
```

After Claude Code finishes, restart it once so the new MCP server is picked up.

### Option B · Manual install

<details>
<summary>Click to expand manual steps</summary>

**1. Clone and install dependencies**

```bash
git clone https://github.com/ChrisPiz/Henge.git
cd Henge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY  (required — runs the 9 frames + the tenth-man)
#   OPENAI_API_KEY     (default embedding provider; cheaper, most devs already have one)
#   VOYAGE_API_KEY     (optional — better quality for Spanish; uncomment EMBED_PROVIDER=voyage)
```

Verify keys:

```bash
python -m henge.server   # should print "✓ keys validated" then wait for stdio
```

**3. Register the MCP server with Claude Code**

```bash
claude mcp add -s user henge \
  "$(pwd)/.venv/bin/python" -- -m henge.server
```

The `-s user` flag makes it available globally (any project). Drop the flag to scope it to the current directory only.

Or, if you prefer editing config by hand, add to `~/.claude.json` (or your client's `mcp.json`):

```json
{
  "mcpServers": {
    "henge": {
      "command": "/absolute/path/to/Henge/.venv/bin/python",
      "args": ["-m", "henge.server"]
    }
  }
}
```

Restart Claude Code (or your MCP client). Verify with `claude mcp list` — `henge` should report `✓ Connected`.

**4. Optional: add a `/decide` slash command**

```bash
mkdir -p ~/.claude/commands
cat > ~/.claude/commands/decide.md <<'MD'
---
description: Invokes Henge — disagreement map of 9 advisors + 1 dissenter.
---
Use the `decide` MCP tool from the `henge` server to analyze: $ARGUMENTS

When the JSON returns: cite `viz_path`, summarize the consensus first,
list the 9 advisors' conclusions, then quote the tenth-man verbatim.
MD
```

**5. Run the tests (optional)**

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

5 critical tests on design invariants + 2 smoke tests + provider-error handling. Suite runs in <5s with mocked SDK calls.

</details>

### Where reports live

Every invocation persists to `~/.henge/reports/<id>/` with both a `report.html` (the editorial visualization) and a `report.json` (raw data: question, context, 10 responses, distances, summary, cost). A browseable `index.html` lists every past run, newest first.

Override the location with `HENGE_REPORTS_DIR=/path/to/dir` in your `.env` if you want them elsewhere. There is no auto-purge — you decide when to clean up.

---

## Core principle

Henge does not blindly contradict.

If there is no real consensus, forcing disagreement only adds noise.

The tenth-man activates when the system detects meaningful alignment between the other frames.

---

## Available tool

### `henge_analyze`

Runs a structured dissent analysis over a user question.

**Input:**

```json
{
  "question": "Should I hire someone or keep working alone in my business?",
  "context": "Monthly revenue is CLP 2.4M. Hiring would cost around CLP 500K/month. I have family expenses and limited runway."
}
```

**Output:**

```json
{
  "consensus_strength": "weak",
  "most_aligned_frame": "first-principles",
  "most_divergent_frame": "analogical",
  "tenth_man_activated": false,
  "verdict": "No strong consensus detected. Do not force dissent.",
  "recommendation": "Validate whether the operational bottleneck is truly limiting revenue before hiring."
}
```

---

## Cognitive frames

Each advisor runs a distinct reasoning frame. The point isn't to add ten voices — it's to get ten *different* angles on the same question and surface where they agree and where they don't.

- **empirical** — Quantifies with numbers, base rates and evidence. Cites magnitudes and sources, marks every unverified premise as `[supuesto]`. Refuses to speculate when there are no real data.

- **historical** — Cites 2–3 analogous cases from the past and extracts the pattern. Asks "what happened the last three times someone tried this?" Stops first-principles reasoning when precedent is abundant.

- **first-principles** — Reduces the problem to its physical, economic or logical atoms and rebuilds without assuming the conventional approach. Asks "what would have to be true for the standard solution to be optimal?"

- **analogical** — Pulls the *functional mechanism* from another domain — biology, military strategy, chess, finance, complex systems — and adapts it. Avoids surface metaphors; demands structural isomorphism.

- **systemic** — Maps feedback loops, second- and third-order effects. Asks "what happens if everyone does this?" and "who changes their behavior in response?" Refuses to reason in partial equilibrium.

- **ethical** — Crosses the deontological lens (rights, dignity, promises) with the consequentialist one (outcomes at 1, 5, 10 years). Names the tension between them when it exists instead of ducking it.

- **soft-contrarian** — Accepts the question and reframes one silent assumption. *"Yes, but consider that X isn't necessarily true if Y."* Surgical nuance, not opposition. Won't say "it depends".

- **radical-optimist** — Lights the 10× upside scenario. Asks "what unlocks if everything goes right?" and looks for asymmetric bets (capped downside, uncapped upside). Names risks briefly; doesn't dwell on them.

- **pre-mortem** — Assumes the decision already failed at 12 months and describes why. Lists concrete operational failure modes ranked by likelihood, not severity. Diagnoses; doesn't recommend mitigations.

- **tenth-man** — Reads the other nine and is *required* to dissent. Steel-mans the contrarian view: accepts observable facts, attacks shared interpretations, builds the most coherent counter-case it can find. Sounds convincing on purpose — that's its job, not a signal it's right.

---

## Compared to simple prompting

| Approach | Problem | Henge |
|----------|---------|-------------|
| Single prompt | One confident answer | Multiple frames |
| Multi-agent debate | Often noisy | Measures agreement |
| Devil's advocate | Always contradicts | Conditional dissent |
| Tenth-man rule | Fixed opposition | Data-driven dissent |

---

## Use cases

- Founder decisions
- Product strategy
- Hiring and investment decisions
- Risk analysis
- Agent orchestration
- Pre-mortems
- High-uncertainty choices

---

## Not goals

Henge is not:

- a chatbot
- a debate simulator
- a generic agent framework
- a replacement for judgment

It is a focused reasoning tool for structured disagreement.

---

## Example

User asks:

> Should I hire someone for my 3D printing business?

Henge may produce:

> The nine frames converge around caution:
> validate demand before creating a fixed cost.
>
> The tenth-man challenges that:
> waiting for perfect validation may be the mechanism that keeps the founder stuck in operational work.
>
> Final verdict:
> externalize first, measure impact, then convert to a fixed role only if time liberated becomes measurable revenue.

---

## Roadmap

- MCP tool schema refinement
- Consensus strength scoring
- Dissent impact scoring
- HTML/PDF report export
- Embedding-based disagreement map
- Claude Desktop / Cursor usage examples

---

## License

MIT
