# Henge · Limits

This document is a list of things Henge **does not** measure, validate, or
claim. It is referenced from the README and the WHITEPAPER and exists so that
the marketing surface stays honest about what's load-bearing and what isn't.

If you find a limitation that belongs here and isn't, open an issue.

---

## 1 · Embedding distance is not propositional disagreement

Henge measures the cosine distance between text embeddings. Two responses can
phrase the **same** conclusion in different vocabulary and embed far apart.
Two responses can phrase **opposing** conclusions in similar style and embed
close. The embedding distance reflects lexical-semantic similarity, not
logical conflict.

Implication: a high CFI is necessary but not sufficient for "the dissent
contradicts the consensus on a substantive premise". Read the tenth-man
response itself before acting. CFI is a triage signal, not a conclusion.

## 2 · Embeddings are style-sensitive

Even with `text-embedding-3-large` (the v0.6 default, ~15-25% better Spanish
recall than `-3-small`), embeddings encode register and phrasing alongside
meaning. Frames that share a stylistic opener ("considering …", "the data
shows …") land closer to one another than their content alone would imply.
This biases the centroid toward shared vocabulary rather than shared
reasoning.

The v0.6 cross-lab routing (gpt-5 + Anthropic frames) intentionally produces
*different* stylistic registers across the nine, so this bias is partially
attenuated — but not eliminated.

Mitigation: Voyage `voyage-3-large` is opt-in (deprecation candidate for
v0.7); semantic-only models (e.g. instructor-style) are on the roadmap.

## 3 · Two-lab cognitive diversity, not three

v0.6 routes 6 of the 9 frames to OpenAI gpt-5 and 3 to Anthropic
(2× Sonnet 4.6 + 1× Opus 4.7), with audit roles (adversarial scoping,
meta-frame, tenth-man informed, claim verification) crossing to OpenAI by
design. This is a real reduction of the v0.5 single-provider concern: the
synthesis lab and the audit lab no longer share refusal patterns or RLHF
objectives.

What's still residual:

- **Two labs, not three.** Gemini and local models are not yet in the pool.
  Adding a third lab — particularly for audit roles — would further reduce
  the synthesis-vs-audit correlation.
- **Same-era public corpora.** gpt-5 and Anthropic frontier models share
  most of their pretraining-era public web text and code. Cross-lab routing
  reduces same-corpus overfit but does not eliminate it.
- **Prompt induction still dominates.** Most of the structural diversity
  delivered by Henge still comes from the nine fixed system prompts, not
  from model heterogeneity. Different prompts on different backbones are
  better than different prompts on one backbone, but the prompt is still
  doing most of the work.

Status: cross-provider routing shipped in v0.6. Three-lab routing remains
on the roadmap for v0.7+.

## 4 · The tenth-man is not fully independent of the nine

v0.6 splits the tenth-man into two:

- **Blind (Opus 4.7)** — receives the question and context but **not** the
  nine. Anticipates a plausible consensus and steel-mans against it.
  Distance metric uses this output's embedding.
- **Informed (gpt-5, cross-lab)** — receives the nine + the blind output
  and returns structured reconciliation (`what_holds` / `what_revised` /
  `what_discarded`).

The blind half removes the self-fulfilling-prompt issue from the distance
metric — it never sees the nine, so its position cannot be a reaction to
them. The informed half is *deliberately* reactive: its job is to audit
whether the blind dissent survives contact with what the nine actually
said. CFI is now computed against the blind output only; the informed output
is editorial, not metric.

This is still not a perfect independence guarantee — the blind tenth-man is
prompted to disagree with whatever consensus the nine *would have reached*,
which is a softer version of the same self-fulfilment. But it is materially
stronger than v0.5's single-tenth-after-nine design.

## 5 · MDS at N=10 is a chart, not analysis

Classical MDS over 10 points in 1024+ dimensions is illustrative. The 2-D
projection helps a human read the relative geometry; it does not justify any
quantitative claim about cluster structure beyond what the raw cosine
distances say. Distances reported by Henge are computed in the original
embedding space — the MDS layer is purely for visualisation.

## 6 · Verdict thresholds are pre-registered, not optimised

`TIGHT_SIGMA = 0.03`, `CFI_K = 6`, and `CFI_FRAGILE_THRESHOLD = 0.33` are
pre-registered constants (see [`docs/cfi-spec.md`](docs/cfi-spec.md)). They
were chosen as conservative defaults during v0.4–v0.5 development and
inherited unchanged into v0.6. They have not been calibrated against a
ground-truth benchmark because no benchmark exists yet (see §10).

If the planned Henge-50 benchmark suggests different thresholds, they will
move in a new spec version; reports stamped with an older spec remain valid
for that spec only.

## 7 · Cost is computed, not measured at the API edge

The `cost_breakdown` field in every report is derived from the `usage`
fields returned by the Anthropic SDK + OpenAI SDK and the published rate
cards encoded in [`henge/providers/pricing.py`](henge/providers/pricing.py).
If a provider changes prices and we have not bumped the rate card, the
recorded cost is wrong. Every report records a `pricing_version` for
traceability; for billing purposes use the provider's invoice, not Henge's
number.

The v0.6 cost shape is `{anthropic_usd, openai_usd, embedding_usd,
total_usd, pricing_version, by_phase}`. The legacy v0.5 lookup keyed on
raw SDK strings missed every OpenAI call (silent 0.0); the v0.6 rewrite
in `henge.providers.pricing.build_cost_breakdown` fixes this. The legacy
`henge.pricing.total_cost` module is kept for back-compat through v0.7.

The embedding cost line item is currently a **floor** — embedding token
counts are not yet fully propagated from the embed module. The total
understates the true embed cost by ~$0.0005/run. Backlog.

## 8 · No reproducibility guarantee across embedding-model versions

Provider embedding models are versioned, and providers occasionally roll
silent updates. The `embed.model` field in each report records the model name
the run was issued against, but cannot guarantee bit-identity over time. Two
reports run against `text-embedding-3-large` six months apart may differ by
a small amount even with `temperature=0` everywhere. The v0.5 → v0.6
default migration from `-3-small` to `-3-large` was uncalibrated and will
shift absolute CFI values; bin assignment is σ-based and provider-agnostic
so the verdict surface is mostly stable.

## 9 · No privacy guarantees on persisted reports

Reports are written in plain JSON to `~/.henge/reports/`. They include the
full question and context (which may contain personal financials, employment
information, family details, etc.). There is no encryption, no auto-redaction,
and no TTL. If you do not want a question persisted, do not run it through
Henge — there is currently no opt-out path short of `rm` after the fact.

A no-persistence flag is on the roadmap for sensitive decisions.

## 10 · No outcome-validated benchmark yet

Henge's central marketing claim — that forced steel-man dissent surfaces
failure modes single-LLM review misses — is **not yet validated against
ground truth**. The Henge-50 benchmark (50 historical decisions with known
outcomes, scored against Henge's tenth-man output) was originally planned
for v0.6; v0.6 prioritised cross-lab routing instead, so Henge-50 is now
slated for a future release. Until then, treat Henge as a structural tool
with a reproducible metric, not a validated instrument of decision quality.

## 11 · Structural metric, not normative

CFI describes the relative position of the dissenter in the embedding
geometry of the ten responses. It does not measure:

- whether the consensus is correct
- whether the dissent is well-founded
- whether the user should follow the consensus or the dissent
- the magnitude of the stakes
- the reversibility of the decision

A high CFI on a low-stakes reversible decision is less actionable than a low
CFI on a one-way-door decision. Henge does not encode that distinction;
the user must.
