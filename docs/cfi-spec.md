# CFI · Consensus Fragility Index

**Pre-registered specification — Henge v0.6.0**
*Last updated: 2026-05-02 · Spec version: `cfi/2026-05-01` (unchanged from v0.5)*

This document is the canonical definition of the Consensus Fragility Index used
by Henge. The thresholds, normalization constant, and bin assignments here are
**pre-registered** — they are fixed at the time of release and any change requires
a new spec version, a CHANGELOG entry, and a `cfi_spec_version` field in the JSON
schema. Reports written under different spec versions are not directly
comparable.

The point of pre-registration is simple: it removes the freedom to retro-fit
thresholds after looking at runs. Without it, "consensus is aligned-fragile"
becomes a moving target.

## 1 · Inputs

Henge embeds the 10 advisor responses with a single embedding model and computes
a centroid over the **n successful frames** (8 or 9 in practice; the tenth-man
is excluded by construction). For each advisor `i ∈ {0, …, 9}`, the cosine
distance to the centroid is

```
d_i = 1 − cos(advisor_i, centroid_n)
```

CFI consumes:

- `d_tenth = d_9` — the tenth-man's distance to the centroid.
- `D_frames = {d_0, …, d_{n−1}}` — the distances of the n successful frames.

## 2 · Index

Let `μ = mean(D_frames)` and `σ = stddev(D_frames)` (population stddev, divisor
`n`, not `n−1`).

```
σ_floor = max(σ, 1e-6)
CFI_raw = (d_tenth − μ) / (k · σ_floor)
CFI     = clamp(0, 1, CFI_raw)
```

with `k = 6.0` fixed. The choice of `k` is justified informally: at six standard
deviations above the cluster mean, the dissenter has pushed far enough that
the consensus cannot absorb it without restructuring. We do not claim this is
optimal — it is pre-registered as a working baseline that future versions may
re-calibrate against the planned Henge-50 benchmark.

`σ_floor` exists only to avoid division by zero on degenerate clusters where
all 9 frames produce textually identical embeddings; in that case the floor of
`1e-6` rounds CFI to 1.0 for any non-negligible tenth-man distance, which is
the correct semantics — a perfectly tight cluster is maximally fragile.

## 3 · Bins

```
if  σ ≥ TIGHT_SIGMA   →  divided
elif CFI < 0.33       →  aligned-stable
else                  →  aligned-fragile
```

with constants:

| Constant                | Value | Source                    |
|-------------------------|-------|---------------------------|
| `TIGHT_SIGMA`           | 0.03  | `henge/viz.py`            |
| `CFI_K`                 | 6.0   | `henge/viz.py`            |
| `CFI_FRAGILE_THRESHOLD` | 0.33  | `henge/viz.py`            |

`TIGHT_SIGMA = 0.03` was retained from v0.4 σ-only verdict logic for
provider-agnostic compatibility — the spread across 9 frames is more stable
across embedding models than the absolute distances themselves. Two distinct
embedding providers (OpenAI `text-embedding-3-large` ~0.6 baseline, the
v0.6 default; OpenAI `text-embedding-3-small` ~0.6 baseline, legacy v0.5;
Voyage `voyage-3-large` ~0.07 baseline) all classify the same 9-frame
cluster as "tight" under this threshold.

The collapse of `CFI ∈ [0.33, 1.0]` into a single `aligned-fragile` bin is
deliberate. We chose three states — not five, not seven — because tri-state
verdicts are easier to act on, and our N=10 sample size makes finer
discrimination noise. The continuous `CFI` field is also persisted for callers
that need it.

## 4 · Worked example

For tight cluster:

```
D_frames = [0.62, 0.63, 0.64, 0.64, 0.65, 0.65, 0.66, 0.67, 0.68]
μ        = 0.6489
σ        = 0.0186
```

If `d_tenth = 0.66`:

```
CFI_raw = (0.66 − 0.6489) / (6 · 0.0186) = 0.0995
CFI     = 0.0995  →  bin = aligned-stable
```

If `d_tenth = 0.95`:

```
CFI_raw = (0.95 − 0.6489) / (6 · 0.0186) = 2.70  →  clamp → 1.00
CFI     = 1.00     →  bin = aligned-fragile
```

For spread cluster:

```
D_frames = [0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.72, 0.78, 0.85]
σ        = 0.108  ≥ 0.03  →  bin = divided  (overrides CFI)
```

## 5 · Properties

- **Provider-agnostic.** σ and (d − μ) scale together when the embedding
  baseline changes; CFI is unitless.
- **Deterministic** when `temperature=0` (the default for every Anthropic
  call and every gpt-5 call in v0.6; Opus omits the parameter and relies on
  the model's own determinism). Repeated runs of the same question produce
  identical CFI within numerical noise from MDS init (which only affects the
  visual map, not the distance computation) and Opus's reasoning-tier drift.
- **Scoped.** CFI describes the relative position of the dissenter in the
  measured space. It does **not** measure whether the consensus is correct,
  whether the dissent is well-founded, or whether the user should follow
  either. See [LIMITS.md](../LIMITS.md).

## 6 · Validation

Validation against a benchmark of decisions with known outcomes is in
[Henge-50 (planned)](../WHITEPAPER.md#7--validation-plan). v0.5 targeted
v0.6; v0.6 prioritised cross-lab routing; Henge-50 slipped to a future
release. Until then,
CFI is a **structural** metric: it correctly classifies the geometry of
agreement between the 10 responses, but it has not yet been correlated with
decision quality.

## 7 · Change log

| Version          | Date       | Change                                                                                          |
|------------------|------------|-------------------------------------------------------------------------------------------------|
| `cfi/2026-05-01` | 2026-05-01 | Initial pre-registration. Henge v0.5.0.                                                         |
| `cfi/2026-05-01` | 2026-05-02 | No change in v0.6.0. CFI is computed against the **blind** tenth-man only; the informed tenth-man is editorial, not metric. Embedding default migrated from `text-embedding-3-small` to `text-embedding-3-large`; bin assignment is σ-based and remains stable. |
