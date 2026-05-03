# Henge · Structured Dissent Protocol

**Whitepaper v0.6.0 · 2026-05-02**
*Author: ChrisPiz · License: MIT*

---

## Abstract

LLM-based decision aids tend to converge prematurely. A single model trained
on the same corpus reinforces the same priors, and even multi-agent debate
setups frequently devolve into noisy consensus theater. We describe **Henge**,
an MCP server that runs a fixed protocol — the Structured Dissent Protocol
(SDP) — over a single decision question: nine cognitive frames in parallel,
routed across two model families by design (Anthropic + OpenAI), followed by
a *dual* tenth advisor — a blind dissenter trained to anticipate consensus
without seeing it, and an informed cross-lab auditor that reconciles the
blind output against the nine. We describe the metric used to summarise the
geometry of agreement (the Consensus Fragility Index, CFI), the runtime
decisions we pre-register to make runs reproducible, and the limits we
declare. We do not claim Henge produces correct decisions; we claim it
surfaces failure modes that single-LLM review systematically misses, and
that cross-lab routing catches the case where the synthesis lab hallucinates
into consensus. Validation against a benchmark of decisions with known
outcomes (Henge-50) remains planned for a future release.

## 1 · Problem

Single-LLM decision review fails for three structural reasons:

1. **Convergence.** A model trained on coherent text predicts coherent
   answers. Asked twice, it tends to repeat itself rather than disagree.
2. **Sycophancy.** Reinforcement-learning-from-human-feedback rewards
   agreement, so models trained that way are biased toward whatever priors the
   user has already exposed.
3. **Performative dissent.** Asking a model "what could go wrong?" produces
   an enumeration of risks that reads like dissent but is structurally a
   continuation of the consensus — the same model arguing with itself, in the
   same vocabulary, under the same priors.

Multi-agent debate frameworks address (1) but typically not (2) or (3): the
agents share a backbone, the debate moderator selects for consensus, and the
"contrarian" agent is a personality skin on the same model.

## 2 · Method · Structured Dissent Protocol (SDP)

SDP runs six phases. v0.6 splits work across two labs (Anthropic + OpenAI)
on purpose: synthesis tasks stay in Anthropic, audit tasks cross to OpenAI,
so that when the synthesis lab hallucinates the auditor catches it.

```
Phase 1 · scoping (base + adversarial cross-lab)
   Haiku 4.5 reads the question and returns 4–6 base clarifying questions
   covering quantitative inputs, constraints, and stakeholders. gpt-5
   (cross-lab) then adds 2–4 adversarial questions that challenge
   unexamined assumptions inside the question itself. Each question
   surfaces id / source (scoping|adversarial) / challenges_assumption.
   Optional — the caller may skip with skip_scoping=True.

Phase 2 · meta-frame audit (cross-lab)
   gpt-5 classifies the question along four axes:
     decision_class · urgency · question_quality · meta_recommendation.
   If the question is exploration disguised as decision, a proxy for
   another question, or has fake urgency, SDP short-circuits with status
   `meta_early_exit` and a `suggested_reformulation`, saving the
   ~USD 1.00 cost of the full run.

Phase 3 · canonical context
   Opus 4.7 canonicalises the user's free-form scoping answers into a
   tight executive summary and flags inconsistencies, before the nine
   advisors see them. This stops the nine from each re-deriving the
   user's intent independently.

Phase 4 · nine frames (cross-lab routing)
   The nine frames run in parallel, routed across labs:
     6× gpt-5: empirical, first-principles, systemic, soft-contrarian,
                radical-optimist, pre-mortem
     2× Sonnet 4.6: analogical, ethical
     1× Opus 4.7: historical
   Each frame answers in the language of the question with operational
   CONSTRAINTS (e.g. ≥3 numerical figures with source class for empirical).
   Minimum 8/9 must succeed for the run to proceed.

Phase 5 · consensus + dual tenth man
   Haiku 4.5 reads the nine answers and produces a structured consensus
   document: where the nine converge, internal tensions, net lean.

   Tenth Man (blind) — Opus 4.7, no view of the nine. Anticipates any
   plausible consensus and steel-mans the strongest alternative over four
   required steps: accept observable facts, identify the strongest
   alternative worldview, build it as strong as possible, name the
   conditions under which it defeats the consensus. Distance metric uses
   this output's embedding.

   Tenth Man (informed) — gpt-5 (cross-lab). Sees the nine + the blind
   and returns structured reconciliation: what_holds (blind survives
   audit), what_revised (blind partially correct, adjusted), what_discarded
   (blind fails). Surfaces consensus-vs-bias separation explicitly.

Phase 6 · claim verification + structure measurement
   Sonnet 4.6 extracts falsifiable claims (factual / prescriptive / causal)
   from the consensus. gpt-5 (cross-lab) verifies each claim against the
   nine frame outputs and rates support_strength: strong / moderate /
   weak / unsupported. Hallucinated consensus claims surface in red as
   `unsupported`.

   All advisor responses are embedded with text-embedding-3-large.
   Cosine distance to the centroid of the n successful frames is computed
   in original embedding space; classical MDS produces a 2-D scatter for
   visualisation only. CFI is computed from these distances (see §5).
```

## 3 · Why steel-man, not devil's advocate

A devil's advocate argues against the consensus opportunistically. The
quality of the argument is incidental; the role is to disturb. Henge's tenth
man does the opposite. The role is *structural* — assigned regardless of how
much the nine agree — and the *method* is steel-manning: build the strongest
version of the opposing case, grounded in the best precedents and cleanest
reasoning available. If the user walks away from a Henge report disagreeing
with the dissent, the test is not whether the dissent was wrong; it is whether
the user defeated the strongest possible version of it.

## 4 · Pre-registered runtime decisions

The following are **pre-registered** in v0.6.0 and do not change without a
spec version bump and CHANGELOG entry. They are recorded in every persisted
report under `runtime`. Routing lives in `henge/config/frame_assignment.py`.

| Decision                          | Value                                                                |
|-----------------------------------|----------------------------------------------------------------------|
| Temperature (Sonnet/Haiku/gpt-5)  | `0`                                                                  |
| Temperature (Opus)                | omitted — Opus 4.7 rejects the parameter (extended thinking on by default; see `henge.agents.MODELS_WITHOUT_TEMPERATURE`) |
| Frame routing                     | 6× `openai/gpt-5` · 2× `anthropic/sonnet-4-6` · 1× `anthropic/opus-4-7` |
| Frame assignment                  | empirical, first-principles, systemic, soft-contrarian, radical-optimist, pre-mortem → gpt-5 · analogical, ethical → Sonnet 4.6 · historical → Opus 4.7 |
| Tenth-man (blind)                 | `anthropic/opus-4-7`                                                 |
| Tenth-man (informed, cross-lab)   | `openai/gpt-5`                                                       |
| Scoping (base)                    | `anthropic/haiku-4-5`                                                |
| Scoping (adversarial, cross-lab)  | `openai/gpt-5`                                                       |
| Meta-frame audit (cross-lab)      | `openai/gpt-5`                                                       |
| Canonical context                 | `anthropic/opus-4-7`                                                 |
| Consensus synthesis               | `anthropic/haiku-4-5`                                                |
| Claim extraction                  | `anthropic/sonnet-4-6`                                               |
| Claim verification (cross-lab)    | `openai/gpt-5`                                                       |
| Embedding model (default)         | `text-embedding-3-large` (OpenAI)                                    |
| Embedding model (opt-in)          | `voyage-3-large` (Voyage, deprecation candidate for v0.7)            |
| Frame max tokens                  | 4000 (gpt-5 frames also pinned to `reasoning_effort="low"`)          |
| Tenth-man max tokens              | 3500                                                                 |
| Min frames to proceed             | 8 of 9                                                               |
| MDS init seed                     | `random_state=42`, `n_init=4`                                        |

`temperature=0` is chosen because reproducibility is a load-bearing property
of the metric. Stylistic variance from `temperature > 0` would propagate into
the embeddings and into CFI, making same-question runs report different
verdicts. We accept the loss of generative variance in exchange for run-to-run
stability. K-runs mode (shipped in v0.5.x) re-introduces temperature variance
explicitly when statistical distribution of CFI is desired.

**Opus exception.** Opus 4.7 ships with extended thinking enabled by default
and refuses the `temperature` parameter. We omit it for Opus calls and rely
on the model's own determinism for the tenth-man dissent. Reasoning-tier
output is not bit-identical across runs even with `temperature=0`, but the
drift is small enough in practice that the CFI bin is stable in our
reproducibility tests. This is the most explicit deviation from full
reproducibility in v0.6; the Henge-50 benchmark will quantify the drift
directly when it lands.

## 5 · Metric · Consensus Fragility Index (CFI)

The full pre-registration lives in [`docs/cfi-spec.md`](docs/cfi-spec.md). In
short, given the n successful frames' distances `D_frames` and the tenth-man
distance `d_tenth`:

```
μ        = mean(D_frames)
σ        = stddev(D_frames)            (population, divisor n)
σ_floor  = max(σ, 1e-6)
CFI      = clamp(0, 1, (d_tenth − μ) / (k · σ_floor))   k = 6
```

with three bins:

```
σ ≥ 0.03                      → divided
else CFI < 0.33               → aligned-stable
else                          → aligned-fragile
```

`σ ≥ 0.03` overrides CFI because if the nine themselves disagree, there was no
consensus to attack — the tri-state collapses regardless of what the
tenth-man did.

CFI is **structural**, not normative. It describes the geometry of agreement
between ten responses to the same question. It does not measure whether the
consensus is correct, whether the dissent is well-founded, or whether the
user should follow either. See [LIMITS.md](LIMITS.md).

## 6 · Implementation

The MCP server exposes a single tool, `decide(question, context, skip_scoping)`.
Phase 0 (scoping) and Phase 1–4 (run) are folded into the same tool to keep
the integration surface minimal: an MCP-compatible host calls `decide`
without context to get back scoping questions, then calls `decide` again
with the user's answers as context to run the protocol.

Each run persists two artifacts under `~/.henge/reports/{id}/`:

- `report.json` — canonical record. Includes the question, context, every
  advisor's response, distances, embeddings (2-D), CFI, runtime metadata,
  prompts hash, and cost breakdown.
- `report.html` — editorial render of the JSON. Pure presentation; the JSON
  is the source of truth.

A regenerated `index.html` ledger lists every past run.

The full architecture diagram lives in [`DEVELOPER.md`](DEVELOPER.md).

## 7 · Validation plan

This v0.6.0 paper makes a structural claim: *given a pre-registered protocol
with `temperature=0`, the same question produces the same CFI bin across
runs, modulo numerical noise from MDS init and Opus's reasoning-tier drift.*
This is testable, falsifiable, and tested in the test suite.

It does **not** make a claim about decision quality. That requires:

1. **Henge-50 benchmark (planned).** A public CSV of 50 historical decisions
   with known 12-month-or-longer outcomes (e.g. Blockbuster vs. Netflix
   1999, Kodak vs. digital 1995). Run Henge over each, compare consensus
   net-lean and tenth-man dissent against the actual outcome. Report the
   share of runs where the tenth-man surfaced a failure mode that
   subsequently materialised. v0.5 promised this for v0.6; v0.6 prioritised
   cross-lab routing instead, so Henge-50 is now slated for a future release.
2. **Inter-run stability metric.** K=10 runs at `temperature=0.7` of the
   same question, report the distribution of CFI and verdict bin. The
   K-runs flag shipped in v0.5.x and is exposed via `k_runs` /
   `run_temperature` on the `decide` MCP tool.
3. **Human-rater agreement.** Ask three independent reviewers to read the
   tenth-man dissent and the consensus separately, and rate which
   surfaces more actionable risk. Report inter-rater κ.
4. **Cross-lab agreement (planned).** Compare blind-vs-informed tenth-man
   distance and the share of `what_holds` / `what_revised` / `what_discarded`
   buckets across runs as a proxy for synthesis-bias detection.

Until those land, we are explicit: Henge is a structural tool with a
falsifiable metric, not a validated decision-quality instrument.

## 8 · Related work

- **Klein, Gary. "Performing a Project Premortem." HBR (2007).** The
  pre-mortem frame in Henge derives from Klein. Klein's protocol is a
  one-pass exercise; SDP folds it into nine parallel lenses.
- **Devil's advocate (Vatican, since 1587).** Procedural dissent role; the
  steel-man method differs in requiring constructive opposition rather than
  opportunistic contradiction.
- **Tenth Man rule (Mossad / IDF, post-1973 Yom Kippur intelligence
  failure).** Where nine analysts agree, a tenth must be assigned to argue
  the opposite. Henge adopts the role; the steel-man method is the
  contribution.
- **Multi-agent debate (Du et al., 2023; Liang et al., 2023).** Debate-style
  setups with model agents. Henge differs in (a) parallel rather than
  iterated, (b) measured rather than narrated, and (c) a single dissenter
  with a fixed structural role rather than a moderator selecting for
  agreement.
- **Cynefin (Snowden, 2007).** Cognitive framework for sense-making across
  problem domains. Cynefin is interpretive; SDP is procedural.

## 9 · Open questions

1. **Frame selection.** The nine frames are fixed. Adaptive frame selection
   per question would likely improve signal but trade away the
   pre-registration property. Roadmap.
2. **Three-lab diversity.** v0.6 splits across Anthropic + OpenAI but does
   not yet include Gemini or local models. Adding a third lab to the frame
   pool — particularly for the audit roles — would further reduce the
   synthesis-vs-audit correlation. Roadmap (v0.7+).
3. **Embedding model effect on CFI.** σ-based bin assignment is provider-
   agnostic, but absolute CFI values shift between models. The v0.5 → v0.6
   migration to `text-embedding-3-large` was uncalibrated; we have not yet
   characterised this drift quantitatively.
4. **Blind-tenth self-fulfilment.** The blind tenth-man does not see the
   nine but is explicitly asked to disagree with whatever consensus they
   would have reached. Some component of `d_tenth` is therefore
   prompt-induced rather than discovery. The informed tenth-man (v0.6)
   partially mitigates by audit-rating the blind output against the actual
   nine; documented in LIMITS.md.
5. **Cross-lab same-priors residual.** gpt-5 and Anthropic models share
   pretraining-era public corpora and similar RLHF objectives. Cross-lab
   routing reduces same-corpus overfit but does not eliminate it.

## 10 · Appendix · prompt set

The ten prompts are bundled in `henge/prompts/*.md` and loaded once at server
startup. The SHA-256 prefix of the ordered concatenation is exposed as
`henge.agents.PROMPTS_HASH` and persisted in every report so future readers
can verify which prompt set produced any given record.

Current `prompts_hash` for v0.6.0: see `henge.agents.PROMPTS_HASH` at runtime.
