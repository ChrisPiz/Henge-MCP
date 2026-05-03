# Changelog

All notable changes to Henge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semantic versioning once it reaches 1.0. Pre-1.0 versions may
introduce additive fields freely; field removals or breaking changes are
documented under **Removed** with migration notes.

---

## [0.6.0] - 2026-05-02

### Added
- **Provider abstraction layer** (`henge/providers/`): unified async interface
  for Anthropic + OpenAI via `complete(canonical_id, req)`. Canonical model ids
  (`anthropic/sonnet-4-6`, `anthropic/opus-4-7`, `openai/gpt-5`) are the
  contract; raw SDK strings live only inside each provider.
- **Multi-model 9 frames** (`henge/config/frame_assignment.py`): the 9
  cognitive frames now route per-frame instead of all using Sonnet. Mix:
  2× Sonnet 4.6 (analogical, ethical) + 1× Opus 4.7 (historical) +
  6× gpt-5 (empirical, first-principles, systemic, soft-contrarian,
  radical-optimist, pre-mortem). Cross-lab by design.
- **Operational CONSTRAINTS** in all 9 frame prompts: each frame now ends
  with a MUST/CANNOT/Output-format block specifying verifiable structure
  (e.g. "cite at least 3 numerical figures with a source class"). Spanish
  questions get Spanish section headers.
- **Adversarial scoping** (`henge/scoping.py::run_scoping`): Haiku 4.5
  produces base scoping questions, then gpt-5 cross-lab adds 2-4
  adversarial questions that challenge unexamined assumptions in the
  question itself. The scoping response surfaces `id`/`source`/
  `challenges_assumption` per question.
- **Canonical context** (`henge/scoping.py::finalize_context`): Opus 4.7
  canonicalizes the user's free-form scoping answers into a tight
  executive summary + flags inconsistencies, before the 9 advisors see it.
- **Meta-frame audit** (`henge/meta_frame.py`): gpt-5 cross-lab classifies
  the question along 4 axes (decision_class, urgency, question_quality,
  meta_recommendation) BEFORE the 9 advisors run. If the question is
  exploration disguised as decision or a proxy for the real question,
  short-circuits with status `meta_early_exit` + `suggested_reformulation`,
  saving ~$1.00/run.
- **Tenth-man dual** (`henge/tenth_man.py`): replaces the single Opus
  tenth-man with two:
    - **Blind** (Opus 4.7) — no view of the 9, anticipates any plausible
      consensus and produces the strongest pure dissent. Distance metric
      uses this output's embedding.
    - **Informed** (gpt-5, cross-lab) — sees the 9 + the blind, returns
      structured reconciliation: `what_holds`, `what_revised`,
      `what_discarded`. Surfaces consensus-vs-bias separation explicitly.
- **Claim verification** (`henge/claims.py`): after the consensus is
  synthesized, Sonnet extracts falsifiable claims (factual / prescriptive
  / causal). Then gpt-5 cross-lab verifies each claim against the 9 frame
  outputs and rates support_strength (strong / moderate / weak /
  unsupported). Hallucinated consensus claims surface in red as
  `unsupported`.
- **HTML viz polish**: meta-frame audit card at top, claim-verification
  panel after consensus, tenth-man split into blind + informed cards,
  disagreement map points coloured by model family (anthropic blue /
  openai green), each frame card carries the canonical model id chip.

### Changed
- **Default embedding model**: `text-embedding-3-small` →
  `text-embedding-3-large`. Cost still <USD 0.001/run, ~15-25% better
  Spanish recall.
- **Cost breakdown** (`henge/providers/pricing.build_cost_breakdown`):
  rewrite to use canonical model ids. The legacy v0.5 lookup keyed on
  raw SDK strings missed every OpenAI call (silent 0.0). Now splits
  honestly into `anthropic_usd` / `openai_usd` / `embedding_usd`, plus
  a `by_phase` dict.
- **Schema bump**: `schema_version` "2" → "0.6". `HENGE_VERSION` 0.5.0 →
  0.6.0.
- **Frame token budget**: `FRAME_MAX_TOKENS` 1500 → 4000 + per-call
  `reasoning_effort="low"` for gpt-5 frames. Without this, gpt-5 burned
  the full token budget on internal chain-of-thought and returned empty
  visible content.
- **OPENAI_API_KEY is now required** — gpt-5 powers 6/9 frames, meta-frame,
  tenth-man informed, claim verification, and adversarial scoping. Add it
  to `.env` if migrating from v0.5.

### Cost target
- Estimated USD ~1.00–1.50 per `/decide` run with all v0.6 features
  enabled. Roughly half Anthropic (3 frames + Opus blind + Opus
  canonical + Haiku scoping/consensus + Sonnet claims-extract) and half
  OpenAI (6 frames + gpt-5 informed + meta-frame + adversarial + claim-
  verify + embeddings).

### Architectural rationale (cross-lab)
- **Synthesis tasks stay in Anthropic** — scoping (Haiku), consensus
  (Haiku), canonical context (Opus), claims extraction (Sonnet),
  tenth-man blind (Opus). Same-lab consistency where structure matters.
- **Audit tasks cross to OpenAI** — adversarial scoping, meta-frame,
  tenth-man informed, claim verification. Cross-lab specifically catches
  the case where the synth lab hallucinates: gpt-5 has no output-style
  affinity with Haiku/Sonnet and surfaces orphan claims.

### Deprecated
- Single-model 9-Sonnet configuration (still possible by overriding
  `FRAME_MODEL_MAP`, not the default).
- Legacy `henge.pricing.total_cost` — replaced by
  `henge.providers.pricing.build_cost_breakdown`. The legacy module is
  kept for back-compat until v0.7 cleanup.
- `EMBED_PROVIDER=voyage` is still supported but unused by default.
  v0.7 may remove the Voyage path entirely if no demand surfaces.

### Notes for v0.7+
- `cross_lab_agreement` and `delta_signal` metrics on the tenth-man pair
  (high/medium/low) — needs embeddings of both blind and informed.
- Claim-extraction inline annotation in the consensus body (vs. separate
  panel in v0.6).
- `--force-full-run` MCP flag to bypass meta-frame `reformulate`
  recommendations.
- Feature-flag-aware reading guide.

---

## [0.5.0] — 2026-05-01

The "Validity + paper" release. Adopts a DORA-style hybrid model: rigor
academic mínimo (paper + límites declarados + reproducibilidad) + marketing
surface intacto (Tenth Man, mapa, ritual). All output JSON changes are
**additive with soft deprecation** — existing integrations continue to work.

### Added

- **Consensus Fragility Index (CFI)** — pre-registered scalar 0–1 metric
  surfaced in every report. Spec: [`docs/cfi-spec.md`](docs/cfi-spec.md).
  New summary fields: `cfi`, `cfi_bin`, `mu_9`, `sigma_9`.
- **Real cost accounting** via `henge.pricing`. Reports now include
  `cost_breakdown.{anthropic_usd, embedding_usd, total_usd, pricing_version}`
  derived from the `usage` field returned by the Anthropic SDK. Replaces
  the v0.4 hardcoded `cost_usd = 0.65`.
- **Runtime metadata** in every report under `runtime`: `henge_version`,
  `temperature`, `model_versions`, `embed.{provider, model}`,
  `prompts_hash`, `n_frames_succeeded`, `n_frames_embedded`. Persisted so
  any future reader can reproduce the run.
- **Per-call usage** under `usage.per_advisor`, `usage.scoping`,
  `usage.consensus`, `usage.embedding`. Token counts come straight from
  the SDK.
- **`PROMPTS_HASH`** — SHA-256 prefix over the ordered concat of the 10
  prompt files. Persisted in every report so prompt drift is detectable.
- **WHITEPAPER.md** — v0.5.0 specification of the Structured Dissent
  Protocol (SDP), pre-registered runtime decisions, validation plan.
- **LIMITS.md** — declared list of what Henge does *not* measure, validate,
  or claim. Linked from README and WHITEPAPER.
- **METHODOLOGY.md** — reproducible step-by-step protocol, reproducibility
  envelope, comparison rules between reports.
- **MANIFESTO.md** — the poetic / philosophical framing previously
  embedded in the README.
- **`docs/cfi-spec.md`** — formal CFI specification with worked examples.
- **CI** — GitHub Actions workflow at `.github/workflows/test.yml` running
  pytest on Python 3.11 and 3.12 across push and pull request.
- **6 new tests** — `test_temperature_is_zero`,
  `test_project_mds_excludes_failed_frames`, `test_prompts_hash_stable`,
  `test_cost_breakdown_sums_components`, `test_no_hardcoded_cost_in_logic`,
  `test_compute_cfi_three_bins`.

### Changed

- **`temperature=0` pinned** in every Anthropic call (frames, scoping,
  consensus, tenth-man). Reproducibility > stylistic variance — the
  pre-registered choice is documented in `WHITEPAPER.md` §4.
- **Embedding cache directory** moved from cwd-relative `./.embed_cache`
  to absolute `~/.henge/embed_cache`. The legacy directory is ignored
  with a one-time stderr notice; safe to delete.
- **`run_agents` return shape** is now a list of
  `(frame, response, status, usage)` 4-tuples (was 3-tuple). The fourth
  element is the SDK `usage` dict for `status == "ok"` and `None` for
  `status == "failed"`. Tests using positional indexing (`r[2]`) keep
  working.
- **`generate_questions` and `synthesize_consensus` return shape** are
  now `(value, usage)` tuples instead of bare value. On failure both
  return `(None, None)`.
- **`project_mds(embeddings, n_frames=None)`** accepts variable frame
  counts. When `n_frames < len(embeddings) - 1`, the tenth-man is
  always the last embedding by convention; previous fixed `n_frames=9`
  behaviour is the default when omitted.
- **README.md** rewritten as academic/corporate interface. Manifesto-tone
  content moved to MANIFESTO.md. Link surface added to WHITEPAPER, LIMITS,
  METHODOLOGY, MANIFESTO, and CFI spec.
- **`schema_version`** in `report.json` bumped from `"1"` to `"2"`.
  All v0.4 fields remain present; v0.5 adds the new metadata blocks.
- **`storage.py` ledger label** bumped from `v0.4` to `v0.5`.

### Fixed

- **Embedding bug** (server.py): when 1–2 frames failed (allowed by the
  8/9 minimum), their `"[failed: …]"` stub text was being embedded
  alongside the successful responses, polluting the centroid and silently
  corrupting all distances. v0.5 embeds only the successful frames + the
  tenth-man and maps distances back to a length-10 list with `None` for
  failed slots.
- **`most_divergent_frame` / `closest_frame` lookup** previously used
  `frame_distances.index(value)` which can return the wrong index on tied
  distances. Now uses explicit `(index, value)` pairs.
- **Opus 4.7 temperature rejection.** Initial v0.5 release pinned
  `temperature=0` on every Anthropic call, but Opus 4.7 (reasoning tier
  with extended thinking on by default) refuses the parameter. Henge now
  omits `temperature` for models in `henge.agents.MODELS_WITHOUT_TEMPERATURE`
  (currently `{claude-opus-4-7}`) and falls back to a no-temperature retry
  when an unknown model surfaces a temperature-related API error. Trade-off
  documented in `WHITEPAPER.md` §4 — the Opus tenth-man is no longer
  bit-reproducible, but the CFI bin remains stable in practice.

### Deprecated

These fields stay in the response and report payload through v0.x and will
be removed in v1.0. Migration guidance below; new integrations should use
the v0.5 replacements.

| Deprecated field            | Replacement                          |
|-----------------------------|--------------------------------------|
| `summary.consensus_state`   | `summary.cfi_bin`                    |
| `summary.consensus_fragility` | derive from `cfi_bin` + locale     |
| `cost_usd` (top-level)      | `cost_breakdown.total_usd`           |

The legacy verdict thresholds (`TIGHT_SIGMA = 0.03`, `DISSENT_SIGMA = 3.0`
in `viz.py`) continue to drive `consensus_verdict`. The v0.5 CFI uses the
same `TIGHT_SIGMA` floor for the `divided` bin so the new and old verdicts
agree on tri-state classification within numerical tolerance.

---

## [0.4.0] — 2026-04 (pre-CHANGELOG)

Last v0.4 release. The CHANGELOG was introduced in v0.5; for prior history
see `git log` and the GitHub release notes.
