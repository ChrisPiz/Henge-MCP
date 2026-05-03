"""Henge server. Exposes 1 tool: decide(question, context?).

Boots only after validating Anthropic + embed provider keys with a minimal ping.
Fail-fast on auth so the developer sees a clear error in T+5s instead of T+60s
mid-invocation.
"""
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from . import pricing
from .providers.pricing import build_cost_breakdown
from .agents import (
    OPUS,
    PROMPTS_HASH,
    SONNET,
    TEMPERATURE,
    TENTH_MAN,
    run_agents,
)
from .consensus import HAIKU as CONSENSUS_HAIKU
from .consensus import synthesize_consensus
from .embed import embed_responses, project_mds
from .scoping import HAIKU as SCOPING_HAIKU
from .scoping import (
    CanonicalContext,
    ScopingResult,
    finalize_context,
    generate_questions,
    run_scoping,
)
from .claims import (
    ClaimVerification,
    ENABLE_CLAIM_VERIFICATION,
    extract_claims,
    verify_claims,
)
from .meta_frame import (
    ENABLE_META_FRAME,
    MetaFrameResult,
    evaluate_question_quality,
)
from .storage import make_report_dir, make_report_id, write_index, write_record
from .updater import get_update_status, update_message
from .viz import compute_cfi, consensus_verdict, render

HENGE_VERSION = "0.6.0"

# Load .env from project root regardless of cwd (Claude Code may spawn subprocess elsewhere).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

mcp = FastMCP("henge")


def _validate_keys_at_startup():
    """Ping Anthropic + OpenAI + (optional) Voyage with minimal calls.
    Cost: ~USD 0.0002. Auth failures here = clear error and exit.
    """
    errors = []

    # Anthropic — required
    if not os.getenv("ANTHROPIC_API_KEY"):
        errors.append(
            "ANTHROPIC_API_KEY no está en el environment. "
            "Obtén una en https://console.anthropic.com"
        )
    else:
        try:
            from anthropic import Anthropic
            Anthropic().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
        except Exception as exc:
            errors.append(
                f"ANTHROPIC_API_KEY validación falló: {type(exc).__name__}: {exc}. "
                f"Verifica con `cat .env | grep ANTHROPIC`."
            )

    # OpenAI — required for v0.6 (gpt-5 frames + meta + informed + claim verification)
    if not os.getenv("OPENAI_API_KEY"):
        errors.append(
            "OPENAI_API_KEY no está en el environment. "
            "v0.6 lo necesita para gpt-5 (frames + meta-frame + tenth-man informed + "
            "claim verification) además de embeddings. "
            "Obtén una en https://platform.openai.com/api-keys"
        )
    else:
        try:
            from openai import OpenAI
            client = OpenAI()
            # Probe gpt-5 access via models.retrieve (zero tokens, confirms
            # the key + the model is reachable on this account). A
            # chat.completions ping doesn't work — gpt-5 is a reasoning
            # model and burns the whole max_completion_tokens budget on
            # internal chain-of-thought before producing visible output.
            client.models.retrieve("gpt-5")
        except Exception as exc:
            errors.append(
                f"OPENAI_API_KEY validación falló (gpt-5): {type(exc).__name__}: {exc}. "
                f"Si tu cuenta no tiene acceso a gpt-5, mira el README — v0.6 requiere acceso al modelo frontier."
            )
        else:
            # Embedding probe (Phase 7 default = 3-large)
            try:
                client.embeddings.create(
                    model="text-embedding-3-large",
                    input=["ping"],
                )
            except Exception as exc:
                errors.append(
                    f"OPENAI_API_KEY validación falló (embeddings): {type(exc).__name__}: {exc}"
                )

    # Voyage — optional, only when EMBED_PROVIDER=voyage
    if os.getenv("EMBED_PROVIDER", "openai").lower() == "voyage":
        if not os.getenv("VOYAGE_API_KEY"):
            errors.append(
                "VOYAGE_API_KEY no está en el environment (configuraste EMBED_PROVIDER=voyage). "
                "Obtén una en https://voyage.ai (free tier 200M tokens/mes), "
                "o quita EMBED_PROVIDER para usar OpenAI por default."
            )
        else:
            try:
                import voyageai
                voyageai.Client().embed(texts=["ping"], model="voyage-3-large")
            except Exception as exc:
                errors.append(
                    f"VOYAGE_API_KEY validación falló: {type(exc).__name__}: {exc}"
                )

    if errors:
        print("✗ Henge startup falló:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print("✓ keys validated", file=sys.stderr)


async def _compute_cfi_only(client, question, context, temperature):
    """Lightweight K-runs companion: frames + tenth-man + embed + CFI.

    Skips consensus synthesis, HTML rendering, and persistence — those are
    only run for the primary K-runs iteration. Returns either a CFI block
    plus per-run cost, or an error dict.
    """
    try:
        agent_result = await run_agents(question, context, temperature=temperature)
        nine = agent_result["nine"]
        blind = agent_result["blind"]
        blind_status = "ok" if blind.opus_usage else "failed"
        results = list(nine) + [(TENTH_MAN, blind.text, blind_status, blind.opus_usage)]
    except RuntimeError as exc:
        return {"error": "agents_failed", "reason": str(exc)}

    success_indices = [i for i, (_, _, s, _) in enumerate(results) if s == "ok"]
    texts_for_embed = [results[i][1] for i in success_indices]

    embed_result = embed_responses(texts_for_embed)
    if not embed_result["ok"]:
        return {"error": "embed_failed", "reason": embed_result.get("reason")}

    n_compact_frames = sum(1 for _, _, s, _ in results[:9] if s == "ok")
    proj = project_mds(
        embed_result["embeddings"],
        n_frames=n_compact_frames,
        embeddings_extra=embed_result.get("embeddings_extra"),
    )

    distances_list = [None] * 10
    for compact_i, orig_i in enumerate(success_indices):
        distances_list[orig_i] = proj["distance_to_centroid_of_9"][compact_i]

    frame_distances_compact = [d for d in distances_list[:9] if d is not None]
    cfi_block = compute_cfi(distances_list[9], frame_distances_compact)

    advisor_usages_cfi = [u for _, _, _, u in results[:9]]
    cost = build_cost_breakdown(
        advisor_usages=advisor_usages_cfi,
        blind_usage=blind.opus_usage,
        informed_usage=None,
        meta_usage=None,
        canonical_usage=None,
        scoping_haiku_usage=None,
        scoping_adversarial_usage=None,
        consensus_usage=None,
        claims_extract_usage=None,
        claims_verify_usage=None,
        embedding_model=embed_result["model"],
        embedding_input_tokens=0,
    )

    return {
        "cfi": cfi_block["cfi"],
        "cfi_bin": cfi_block["cfi_bin"],
        "mu_9": cfi_block["mu_9"],
        "sigma_9": cfi_block["sigma_9"],
        "tenth_distance": distances_list[9],
        "cost_usd": cost["total_usd"],
    }


@mcp.tool()
async def decide(
    question: str,
    context: str | None = None,
    skip_scoping: bool = False,
    k_runs: int = 1,
    run_temperature: float | None = None,
) -> dict:
    """Disagreement map: 9 advisors + 1 mandatory dissenter (tenth man).

    TWO PHASES:

    Phase 1 — Scoping (default when no context is passed):
    Returns {status: "needs_context", questions: [...]} with 4-7 concrete,
    domain-tailored questions. The caller (Claude Code) must present them to
    the user before proceeding. Without rich context, advisors speculate or
    stay generic.

    Phase 2 — Run (when context is present or skip_scoping=True):
    Runs the 9 advisors in parallel + the tenth man, returns a 2D map with
    distances, the dissenter's literal quote, and opens the HTML in a browser.

    K-RUNS MODE (v0.5):

    When ``k_runs > 1`` and ``run_temperature > 0`` are supplied, Henge runs
    the pipeline K times sequentially with the given temperature, collecting
    a CFI distribution (mean / stddev / 95% CI) so callers can characterise
    run-to-run variance. Only the first run produces a persisted HTML report;
    subsequent runs contribute their CFI to the distribution only. K-runs
    requires ``context`` (or ``skip_scoping=True``) — scoping is skipped.

    Args:
        question: The decision or judgment question.
        context: User answers to the scoping questions, or rich initial
                 context. If empty, phase 1 is entered.
        skip_scoping: If True, skip scoping and run directly. Useful when the
                      caller decides the question already has enough context.
        k_runs: Number of pipeline executions. Default 1 (reproducible at
                temperature=0). Values > 1 require run_temperature.
        run_temperature: Override temperature for K-runs > 1. Required when
                         k_runs > 1. Ignored when k_runs == 1.

    Returns:
        Phase 1: {status: "needs_context", questions: [...], note, next_call_hint}
        Phase 2 (K=1): {viz_path, frames, tenth_man, summary, cost_breakdown, cost_usd}
        Phase 2 (K>1): same as K=1 plus summary.k_runs_distribution
    """
    if not question or not question.strip():
        return {"error": "empty_question", "reason": "Provide a non-empty question."}

    if k_runs < 1:
        return {"error": "invalid_k_runs", "reason": "k_runs must be >= 1."}
    if k_runs > 1:
        if run_temperature is None or run_temperature <= 0:
            return {
                "error": "k_runs_requires_temperature",
                "reason": (
                    "k_runs > 1 needs run_temperature > 0 to produce variance. "
                    "Pass e.g. run_temperature=0.7. Default temperature=0 is "
                    "deterministic and a K-runs distribution would be degenerate."
                ),
            }
        if not context and not skip_scoping:
            return {
                "error": "k_runs_requires_context",
                "reason": (
                    "k_runs > 1 mode skips scoping. Pass context= with the "
                    "answers already collected, or skip_scoping=True."
                ),
            }

    client = AsyncAnthropic()

    scoping_usage: dict | None = None
    if not context and not skip_scoping:
        scoping_result = await run_scoping(question)
        if scoping_result.haiku_usage:
            scoping_usage = dict(scoping_result.haiku_usage)
        if scoping_result.gpt5_usage:
            if scoping_usage is None:
                scoping_usage = dict(scoping_result.gpt5_usage)
            else:
                scoping_usage = {
                    "model": scoping_usage["model"],
                    "input_tokens": scoping_usage["input_tokens"] + scoping_result.gpt5_usage["input_tokens"],
                    "output_tokens": scoping_usage["output_tokens"] + scoping_result.gpt5_usage["output_tokens"],
                }
        if not scoping_result.questions:
            return {
                "status": "scoping_failed",
                "reason": "Could not generate scoping questions. Pass skip_scoping=True or provide context to proceed.",
            }
        return {
            "status": "needs_context",
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "source": q.source,
                    "challenges_assumption": q.challenges_assumption,
                }
                for q in scoping_result.questions
            ],
            "adversarial_count": scoping_result.adversarial_count,
            "note": (
                "Present these questions to the user in a numbered message, wait for "
                "their reply, then call decide() again with question + their answers "
                "as context. Adversarial questions (source='adversarial') challenge a "
                "hidden assumption in the original question — surface those distinctly. "
                "ALSO tell the user — at the bottom of the questions message — that "
                "they can skip the questions and run immediately by saying 'skip' / "
                "'omitir' / 'corre ya', in which case call decide again with "
                "skip_scoping=True."
            ),
            "skip_hint": (
                "Optional escape hatch: if you'd rather not answer, just say 'skip' "
                "(or 'omitir' / 'corre ya') and Henge will run the 9 advisors + the "
                "tenth-man on the question as-is."
            ),
            "next_call_hint": f"decide(question={question!r}, context='<user answers formatted>')",
            "skip_call_hint": f"decide(question={question!r}, skip_scoping=True)",
        }

    primary_temperature = run_temperature if k_runs > 1 else TEMPERATURE

    # Phase 3: when the user provided free-form context, run Opus canonicalization
    # so the 9 frames see a tight executive summary instead of raw user prose.
    # Behind HENGE_ENABLE_CANONICAL_CONTEXT (default true). Falls back to passthrough
    # on flag=false or Opus failure (graceful, see scoping.finalize_context).
    canonical_usage: dict | None = None
    if context and not skip_scoping:
        canonical = await finalize_context(question, context)
        effective_context = canonical.summary
        canonical_usage = canonical.opus_usage
    else:
        effective_context = context

    # Phase 4: meta-frame audit. gpt-5 cross-lab audits the question itself
    # before the 9 advisors spend tokens. If the question is malformed
    # (exploration disguised as decision, or a proxy for the real question),
    # short-circuit with a recommendation. Behind HENGE_ENABLE_META_FRAME.
    meta: MetaFrameResult | None = None
    meta_usage: dict | None = None
    if ENABLE_META_FRAME and effective_context:
        meta = await evaluate_question_quality(question, effective_context)
        meta_usage = meta.gpt5_usage
        if meta.meta_recommendation in ("reformulate", "this-is-not-a-decision"):
            return {
                "status": "meta_early_exit",
                "meta_recommendation": meta.meta_recommendation,
                "decision_class": meta.decision_class,
                "urgency": meta.urgency,
                "question_quality": meta.question_quality,
                "suggested_reformulation": meta.suggested_reformulation,
                "reasoning": meta.reasoning,
                "next_call_hint": (
                    f"decide(question='<reformulated>', context={context!r})"
                    if meta.meta_recommendation == "reformulate"
                    else "This is not a decision question — try /explica or rephrase."
                ),
            }

    try:
        agent_result = await run_agents(question, effective_context, temperature=primary_temperature)
    except RuntimeError as exc:
        return {"error": "agents_failed", "reason": str(exc)}

    nine = agent_result["nine"]
    blind = agent_result["blind"]
    informed = agent_result["informed"]

    # Build the v0.5-shape ``results`` list (9 frames + blind as the tenth slot)
    # so the consensus / embedding / distance / cost code below keeps working
    # unchanged. The blind output drives the distance metric (per Phase 5 spec).
    blind_status = "ok" if blind.opus_usage else "failed"
    results = list(nine) + [(TENTH_MAN, blind.text, blind_status, blind.opus_usage)]

    # results: list of (frame, response, status, usage) tuples, length 10.
    # status "ok" → usage is dict; status "failed" → usage is None.
    successful_frames = [(f, r) for f, r, s, _ in results[:9] if s == "ok"]
    consensus_text, consensus_usage = await synthesize_consensus(client, successful_frames, question)

    # Phase 6: extract falsifiable claims from the consensus, then cross-lab
    # verify each against the 9 advisors. Catches the case where Haiku
    # hallucinated a claim that no advisor actually wrote.
    claim_verifications: list[ClaimVerification] = []
    extract_claims_usage: dict | None = None
    verify_claims_usage: dict | None = None
    if ENABLE_CLAIM_VERIFICATION and consensus_text:
        extracted, extract_claims_usage = await extract_claims(consensus_text)
        if extracted:
            claim_verifications, verify_claims_usage = await verify_claims(
                extracted, successful_frames
            )

    # v0.5 fix: previously embedded ALL 10 responses, including the
    # "[failed: ...]" stub for failed frames. That polluted the centroid and
    # silently corrupted distances when 1-2 frames failed (8/9 minimum allows
    # this). Now: embed only successful frames + the tenth-man, and map
    # distances back to length-10 lists with None for failed slots.
    success_indices = [i for i, (_, _, s, _) in enumerate(results) if s == "ok"]
    texts_for_embed = [results[i][1] for i in success_indices]

    embed_result = embed_responses(texts_for_embed)
    if not embed_result["ok"]:
        return embed_result

    n_compact_frames = sum(1 for _, _, s, _ in results[:9] if s == "ok")
    proj = project_mds(
        embed_result["embeddings"],
        n_frames=n_compact_frames,
        embeddings_extra=embed_result.get("embeddings_extra"),
    )

    distances_list = [None] * 10
    coords_list = [None] * 10
    for compact_i, orig_i in enumerate(success_indices):
        distances_list[orig_i] = proj["distance_to_centroid_of_9"][compact_i]
        coords_list[orig_i] = proj["coords_2d"][compact_i]

    # Phase 8: v0.6 cost_breakdown using canonical model ids — captures
    # OpenAI gpt-5 spend (frames + informed + meta + adversarial scoping +
    # claim verification) that the legacy anthropic-only path missed.
    advisor_usages = [u for _, _, _, u in results[:9]]  # 9 frames only
    # Reconstruct scoping breakdown from the merged scoping_usage dict.
    # When run_scoping returned both Haiku + gpt-5, scoping_usage carries the
    # merged anthropic/haiku-4-5 sum; the gpt-5 portion is lost in the
    # legacy aggregation. For Phase 8 cost honesty, we attribute the merged
    # blob to scoping_haiku and leave scoping_adversarial=None when we can't
    # split it. Phase 3.4's shim returns the merged dict only.
    scoping_haiku_usage = None
    scoping_adversarial_usage = None
    if scoping_usage and scoping_usage.get("model", "").startswith("anthropic/"):
        scoping_haiku_usage = scoping_usage
    cost_breakdown = build_cost_breakdown(
        advisor_usages=advisor_usages,
        blind_usage=blind.opus_usage,
        informed_usage=informed.gpt5_usage,
        meta_usage=meta.gpt5_usage if meta else None,
        canonical_usage=canonical_usage,
        scoping_haiku_usage=scoping_haiku_usage,
        scoping_adversarial_usage=scoping_adversarial_usage,
        consensus_usage=consensus_usage,
        claims_extract_usage=extract_claims_usage,
        claims_verify_usage=verify_claims_usage,
        embedding_model=embed_result["model"],
        embedding_input_tokens=0,
    )
    cost_usd = cost_breakdown["total_usd"]

    # Successful-frame distances only — None entries are failed frames that
    # were not embedded. CFI and verdict use these compact lists.
    frame_distances_compact = [d for d in distances_list[:9] if d is not None]
    successful_frame_pairs = [
        (i, distances_list[i]) for i in range(9) if distances_list[i] is not None
    ]
    if successful_frame_pairs:
        max_frame_idx, max_frame_distance = max(successful_frame_pairs, key=lambda x: x[1])
        min_frame_idx, min_frame_distance = min(successful_frame_pairs, key=lambda x: x[1])
        most_divergent_frame = results[max_frame_idx][0]
        closest_frame = results[min_frame_idx][0]
    else:
        max_frame_idx = min_frame_idx = None
        max_frame_distance = min_frame_distance = None
        most_divergent_frame = closest_frame = None

    tenth_distance = distances_list[9]
    verdict = consensus_verdict(tenth_distance, frame_distances_compact)
    cfi_block = compute_cfi(tenth_distance, frame_distances_compact)

    # viz still expects 3-tuples (frame, response, status). Strip usage.
    results_for_viz = [(f, r, s) for f, r, s, _ in results]

    # viz expects every distance/coord populated; failed frames have None. Fill
    # with safe defaults so the map renders without crashing — failed slots
    # land at the centroid (origin in MDS space) and pick up the cluster mean
    # distance. The frames table still surfaces ``status == "failed"`` so the
    # user sees what actually happened.
    if frame_distances_compact:
        _mean_dist = sum(frame_distances_compact) / len(frame_distances_compact)
    else:
        _mean_dist = 0.0
    distances_for_viz = [d if d is not None else _mean_dist for d in distances_list]
    coords_for_viz = [c if c is not None else [0.0, 0.0] for c in coords_list]

    html = render(
        question=question,
        results=results_for_viz,
        consensus=consensus_text,
        coords_2d=coords_for_viz,
        distances=distances_for_viz,
        provider=embed_result["provider"],
        model=embed_result["model"],
        cost_estimate_usd=cost_usd,
        cfi_data=cfi_block,
        meta_frame=(
            {
                "decision_class": meta.decision_class,
                "urgency": meta.urgency,
                "question_quality": meta.question_quality,
                "meta_recommendation": meta.meta_recommendation,
                "reasoning": meta.reasoning,
            } if meta is not None else None
        ),
        informed=(
            {
                "text": informed.text,
                "what_holds": informed.what_holds,
                "what_revised": informed.what_revised,
                "what_discarded": informed.what_discarded,
            } if informed is not None else None
        ),
        claims=[
            {
                "claim_text": v.claim_text,
                "claim_type": v.claim_type,
                "supporting_frames": v.supporting_frames,
                "contesting_frames": v.contesting_frames,
                "support_strength": v.support_strength,
            }
            for v in claim_verifications
        ],
    )

    n_frames_succeeded = sum(1 for _, _, s, _ in results[:9] if s == "ok")
    embed_input_tokens = sum(len(t) // 4 for t in texts_for_embed)  # rough char/4 estimate

    runtime_meta = {
        "henge_version": HENGE_VERSION,
        "temperature": TEMPERATURE,
        "model_versions": {
            "frames": SONNET,
            "tenth_man": OPUS,
            "scoping": SCOPING_HAIKU,
            "consensus": CONSENSUS_HAIKU,
        },
        "embed": {
            "provider": embed_result["provider"],
            "model": embed_result["model"],
        },
        "prompts_hash": PROMPTS_HASH,
        "n_frames_succeeded": n_frames_succeeded,
        "n_frames_embedded": n_compact_frames,
    }

    usage_block = {
        "per_advisor": [
            {"frame": f, "status": s, "usage": u}
            for f, _, s, u in results
        ],
        "scoping": None,  # scoping returns early before reaching here
        "consensus": consensus_usage,
        "embedding": {
            "provider": embed_result["provider"],
            "model": embed_result["model"],
            "n_texts": len(texts_for_embed),
            "estimated_input_tokens": embed_input_tokens,
        },
    }

    report_id = make_report_id(question)
    report_dir = make_report_dir(report_id)
    payload = {
        "schema_version": "0.6",
        "id": report_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "question": question,
        "context": context,
        "consensus": consensus_text,
        "advisors": [
            {
                "frame": frame,
                "status": status,
                "response": response,
                "distance_to_centroid_of_9": distances_list[i],
                "embedding_2d": coords_list[i],
            }
            for i, (frame, response, status, _) in enumerate(results[:9])
        ],
        "tenth_man": {
            "blind": {
                "text": blind.text,
                "distance": distances_list[9],
                "opus_usage": blind.opus_usage,
                "embedding_2d": coords_list[9],
            },
            "informed": {
                "text": informed.text,
                "what_holds": informed.what_holds,
                "what_revised": informed.what_revised,
                "what_discarded": informed.what_discarded,
                "gpt5_usage": informed.gpt5_usage,
            },
        },
        "consensus_claims": [
            {
                "claim_text": v.claim_text,
                "claim_type": v.claim_type,
                "supporting_frames": v.supporting_frames,
                "contesting_frames": v.contesting_frames,
                "support_strength": v.support_strength,
            }
            for v in claim_verifications
        ],
        "summary": {
            # legacy fields — deprecated, kept until v1.0 for compat
            "tenth_man_distance": tenth_distance,
            "max_frame_distance": max_frame_distance,
            "min_frame_distance": min_frame_distance,
            "most_divergent_frame": most_divergent_frame,
            "closest_frame": closest_frame,
            "consensus_state": verdict["state"],
            "consensus_fragility": verdict["verdict"],
            "n_frames_succeeded": n_frames_succeeded,
            # v0.5 — Consensus Fragility Index (pre-registered, see docs/cfi-spec.md)
            "cfi": cfi_block["cfi"],
            "cfi_bin": cfi_block["cfi_bin"],
            "mu_9": cfi_block["mu_9"],
            "sigma_9": cfi_block["sigma_9"],
        },
        "embed": {
            "provider": embed_result["provider"],
            "model": embed_result["model"],
        },
        "runtime": runtime_meta,
        "usage": usage_block,
        "cost_breakdown": cost_breakdown,
        "cost_usd": cost_usd,  # deprecated alias of cost_breakdown.total_usd, kept for compat
    }

    html_path, json_path = write_record(report_dir, html, payload)
    index_path = write_index()

    try:
        webbrowser.open(f"file://{html_path.absolute()}")
    except Exception:
        pass

    viz_path = str(html_path)

    # Frames truncated to ~300 chars in JSON (full text lives in HTML viz).
    # Tenth-man kept full because Claude must cite it literally per the slash command.
    def _truncate(text, limit=300):
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "..."

    update_status = get_update_status()
    update_available = None
    if update_status and update_status.get("behind", 0) > 0:
        update_available = {
            "behind": update_status["behind"],
            "latest_sha": update_status.get("latest_sha"),
            "current_sha": update_status.get("current_sha"),
            "message": update_message(update_status),
        }

    # K-runs distribution. Primary run already executed at run_temperature; we
    # run K-1 additional iterations sequentially using the same temperature
    # and aggregate the CFI samples into mean / stddev / 95% CI. Sequential
    # rather than parallel to be friendly to provider rate limits — K runs of
    # ~30s each is the expected envelope.
    k_runs_distribution = None
    if k_runs > 1:
        cfi_samples = [cfi_block["cfi"]] if cfi_block.get("cfi") is not None else []
        bin_samples = [cfi_block["cfi_bin"]] if cfi_block.get("cfi_bin") else []
        per_run_costs = [cost_usd]
        per_run_errors = []
        for run_idx in range(1, k_runs):
            extra = await _compute_cfi_only(client, question, context, run_temperature)
            if extra.get("error"):
                per_run_errors.append({"run_idx": run_idx, **extra})
                continue
            if extra.get("cfi") is not None:
                cfi_samples.append(extra["cfi"])
                bin_samples.append(extra["cfi_bin"])
                per_run_costs.append(extra.get("cost_usd", 0.0))

        if cfi_samples:
            n = len(cfi_samples)
            mean = sum(cfi_samples) / n
            sigma = (sum((x - mean) ** 2 for x in cfi_samples) / n) ** 0.5
            ci95_half = 1.96 * sigma / (n ** 0.5) if n > 1 else 0.0
            from collections import Counter
            bin_counts = dict(Counter(bin_samples))
        else:
            mean = sigma = ci95_half = None
            bin_counts = {}

        k_runs_distribution = {
            "k_requested": k_runs,
            "k_completed": len(cfi_samples),
            "run_temperature": run_temperature,
            "cfi_mean": round(mean, 4) if mean is not None else None,
            "cfi_stddev": round(sigma, 4) if sigma is not None else None,
            "cfi_ci95_half_width": round(ci95_half, 4) if ci95_half is not None else None,
            "cfi_samples": [round(x, 4) for x in cfi_samples],
            "bin_distribution": bin_counts,
            "total_cost_usd": round(sum(per_run_costs), 6),
            "errors": per_run_errors,
        }

    return {
        "viz_path": viz_path,
        "report_id": report_id,
        "report_dir": str(report_dir),
        "json_path": str(json_path),
        "index_path": str(index_path),
        "update_available": update_available,
        "viz_note": "Persisted at viz_path (HTML) + json_path (raw data). index_path is the browseable ledger. JSON below contains summaries only (consensus + tenth-man are full).",
        "consensus": consensus_text or "(consensus synthesis failed)",
        "frames": [
            {
                "frame": frame,
                "status": status,
                "distance": (
                    round(distances_list[i], 3) if distances_list[i] is not None else None
                ),
                "summary": _truncate(resp, 300),
            }
            for i, (frame, resp, status, _) in enumerate(results[:9])
        ],
        "tenth_man": {
            "distance": round(tenth_distance, 3) if tenth_distance is not None else None,
            "response": results[9][1],  # full text
        },
        "consensus_claims": [
            {
                "claim_text": v.claim_text,
                "claim_type": v.claim_type,
                "supporting_frames": v.supporting_frames,
                "contesting_frames": v.contesting_frames,
                "support_strength": v.support_strength,
            }
            for v in claim_verifications
        ],
        "summary": {
            # legacy — deprecated, kept until v1.0
            "tenth_man_distance": (
                round(tenth_distance, 3) if tenth_distance is not None else None
            ),
            "max_frame_distance": (
                round(max_frame_distance, 3) if max_frame_distance is not None else None
            ),
            "consensus_state": verdict["state"],
            "consensus_fragility": verdict["verdict"],
            "n_frames_succeeded": n_frames_succeeded,
            "embed_provider": embed_result["provider"],
            "embed_model": embed_result["model"],
            # v0.5 — CFI and runtime metadata
            "cfi": cfi_block["cfi"],
            "cfi_bin": cfi_block["cfi_bin"],
            "mu_9": cfi_block["mu_9"],
            "sigma_9": cfi_block["sigma_9"],
            "henge_version": HENGE_VERSION,
            "prompts_hash": PROMPTS_HASH,
            "k_runs_distribution": k_runs_distribution,
        },
        "meta_frame": (
            {
                "decision_class": meta.decision_class,
                "urgency": meta.urgency,
                "question_quality": meta.question_quality,
                "meta_recommendation": meta.meta_recommendation,
                "reasoning": meta.reasoning,
            }
            if meta is not None
            else None
        ),
        "cost_breakdown": cost_breakdown,
        "cost_usd": cost_usd,  # deprecated alias, kept until v1.0
    }


def main():
    _validate_keys_at_startup()
    # Best-effort version check. Cached for 24h, silent on any failure.
    try:
        msg = update_message(get_update_status())
        if msg:
            print(f"⟳ {msg}", file=sys.stderr)
    except Exception:
        pass
    mcp.run()


if __name__ == "__main__":
    main()
