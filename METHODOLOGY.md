# Henge · Methodology

Step-by-step reproducible protocol. Anyone with the API keys, the same Henge
release, the same `prompts_hash`, and `temperature=0` should be able to
re-run a question and produce a CFI within numerical noise of the original.

---

## 1 · Prerequisites

- Python ≥ 3.11
- `ANTHROPIC_API_KEY` (required — Haiku 4.5, Sonnet 4.6, Opus 4.7)
- `OPENAI_API_KEY` (required in v0.6 — gpt-5 powers 6/9 frames + adversarial
  scoping + meta-frame audit + tenth-man informed + claim verification, plus
  the default `text-embedding-3-large`)
- Optional: `VOYAGE_API_KEY` with `EMBED_PROVIDER=voyage` (deprecation
  candidate for v0.7)
- Henge installed from the repo at a known SHA, or `henge-mcp==X.Y.Z` from
  PyPI (planned)

The simplest path: see the [Quickstart](README.md#quickstart) in the README.

## 2 · Reproducibility envelope

A Henge run is reproducible **only** within this envelope:

| Dimension                | Pinned in v0.6                                                                                  |
|--------------------------|-------------------------------------------------------------------------------------------------|
| `temperature`            | `0` (see WHITEPAPER §4 — Opus omits the parameter)                                              |
| Anthropic models         | `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`                                      |
| OpenAI model             | `openai/gpt-5` (frames + audit roles)                                                           |
| Frame routing            | 6× gpt-5 / 2× Sonnet 4.6 / 1× Opus 4.7 — see `henge/config/frame_assignment.py`                 |
| Tenth-man split          | blind = Opus 4.7 · informed = gpt-5 (cross-lab)                                                 |
| Embedding model          | `text-embedding-3-large` (default, OpenAI)                                                      |
| Prompt set               | identified by `prompts_hash` SHA-256 prefix                                                     |
| MDS init                 | `random_state=42`, `n_init=4` (viz only)                                                        |
| Henge release            | identified by `henge_version` field in report.json                                              |
| CFI spec                 | `cfi/2026-05-01` (unchanged from v0.5; see `docs/cfi-spec.md`)                                  |

If any of these change between runs, the runs are not directly comparable.

## 3 · Protocol

```
Step 1 · Caller sends:
   decide(question="…")

Step 2 · Henge returns:
   {status: "needs_context", questions: [4–7 clarifying questions]}

Step 3 · Caller surfaces those to the user, collects answers, and re-calls:
   decide(question="…", context="…answers…")

Step 4 · Henge runs in this exact order:
   a. scoping (base) → Haiku 4.5
   b. scoping (adversarial, cross-lab) → gpt-5 — adds 2-4 challenge questions
   c. meta-frame audit (cross-lab) → gpt-5 — may short-circuit with
      status="meta_early_exit" + suggested_reformulation
   d. canonical context → Opus 4.7 — tightens scoping answers, flags
      inconsistencies, fans out to all 9 advisors
   e. 9 frames in parallel (asyncio.gather), routed:
        empirical, first-principles, systemic, soft-contrarian,
        radical-optimist, pre-mortem → gpt-5 (reasoning_effort="low")
        analogical, ethical                                → Sonnet 4.6
        historical                                          → Opus 4.7
      All temperature=0; Opus omits the parameter.
      If 8/9 succeed → continue, else abort with structured error.
   f. consensus synthesis from successful frames → Haiku 4.5
   g. tenth-man (blind) over the question + canonical context (NOT the
      nine) → Opus 4.7
   h. tenth-man (informed, cross-lab) over the nine + the blind → gpt-5;
      returns structured what_holds / what_revised / what_discarded
   i. claim extraction from consensus → Sonnet 4.6 (factual /
      prescriptive / causal)
   j. claim verification (cross-lab) → gpt-5; rates each claim
      strong / moderate / weak / unsupported against the nine
   k. embed only the successful frames + blind tenth-man (NOT the failed
      stub texts, NOT the informed tenth-man) using
      text-embedding-3-large
   l. cosine distances to centroid of the n successful frames
   m. classical MDS for the 2-D scatter (visual only)
   n. compute CFI per docs/cfi-spec.md
   o. persist report.json + report.html under ~/.henge/reports/{id}/
   p. return summary JSON to caller

Step 5 · Caller may inspect:
   - viz_path → HTML editorial report
   - json_path → canonical record
   - summary.cfi, summary.cfi_bin, summary.sigma_9, summary.mu_9
   - cost_breakdown.{anthropic_usd, openai_usd, embedding_usd,
                     total_usd, pricing_version, by_phase}
   - runtime.{henge_version, schema_version, model_versions,
              prompts_hash, temperature}
   - meta_frame.{decision_class, urgency, question_quality,
                 meta_recommendation, reasoning, suggested_reformulation?}
   - informed.{what_holds, what_revised, what_discarded}
   - claims[].{text, type, support_strength, evidence_frames}
```

## 4 · Manual reproducibility check

```bash
# Run the same question twice with the same release and key set:
/decide should I take the new job?
# answer the scoping questions
# wait for report

# Open report 1: ~/.henge/reports/{id1}/report.json → record summary.cfi
# Open report 2: ~/.henge/reports/{id2}/report.json → record summary.cfi

# Expected (temperature=0): summary.cfi differs by < 0.05 between runs.
# The only legitimate source of drift is MDS random_state on degenerate
# embeddings — which only affects visualisation, not the cosine distances
# CFI is computed from. If you see a wider drift, file an issue.
```

## 5 · Comparing two reports

Two reports are **directly comparable** iff:

```
report_a.runtime.henge_version    == report_b.runtime.henge_version
report_a.runtime.prompts_hash      == report_b.runtime.prompts_hash
report_a.runtime.model_versions    == report_b.runtime.model_versions
report_a.runtime.embed.model       == report_b.runtime.embed.model
report_a.runtime.temperature       == report_b.runtime.temperature
report_a.cost_breakdown.pricing_version == report_b.cost_breakdown.pricing_version
```

If any field differs, you can still inspect the reports side by side, but
treat the CFI comparison as cross-version, not cross-run.

## 6 · Building a benchmark (planned)

Henge-50 was originally targeted for v0.6; v0.6 prioritised cross-lab
routing and the dual tenth-man instead, so the benchmark slipped. It is
still planned. The shape: a CSV of 50 historical decisions with known
outcomes.

```
id, year, question, context, ground_truth_outcome, source
1, 2000, "Should Blockbuster acquire Netflix for $50M?", "...", "decline (bankruptcy 2010)", "..."
...
```

Validation is not part of v0.6. The methodology, when it lands, will be:

1. Run Henge over each row with `skip_scoping=True` (context is the
   pre-collected case description, no live scoping).
2. Record CFI, consensus net-lean, and tenth-man failure-mode titles.
3. Score:
   - did the consensus net-lean match the historical outcome?
   - did any tenth-man failure mode describe the actual failure that
     materialised?
4. Aggregate over the 50 cases. Report point estimate + 95% CI.

The benchmark will be public so anyone can replicate.

## 7 · Failure modes of the protocol itself

This is not the same as §11 (LIMITS.md) — these are operational failure
modes:

- **Fewer than 8 frames succeed.** Run aborts with a structured error
  identifying the failed frames. Cost is partial (frames that ran are
  billed). Re-run after addressing the API issue.
- **Embed provider down.** Run aborts after frames + tenth-man complete;
  cost is full Anthropic + 0 embedding. The user gets the textual responses
  but no map / no CFI. Future versions will degrade gracefully to
  text-only mode.
- **Same question, different runs, different verdicts.** Should not happen
  with `temperature=0`. If it does:
  1. Confirm `runtime.temperature == 0` in both reports.
  2. Confirm `prompts_hash` matches.
  3. Confirm `model_versions` match.
  4. Confirm `embed.model` matches.
  If all match and verdicts differ, file an issue with both `report.json`
  files attached.
