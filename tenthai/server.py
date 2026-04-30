"""TenthAI MCP server. Exposes 1 tool: decide(question, context?).

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

from .agents import run_agents, TENTH_MAN
from .consensus import synthesize_consensus
from .embed import embed_responses, project_mds
from .scoping import generate_questions
from .storage import make_report_dir, make_report_id, write_index, write_record
from .viz import consensus_verdict, render

# Load .env from project root regardless of cwd (Claude Code may spawn subprocess elsewhere).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

mcp = FastMCP("tenthai")


def _validate_keys_at_startup():
    """Ping Anthropic + embed provider with a minimal call. Cost: ~CLP 0.1.

    Auth failures here = clear error and exit. Auth failures during invocation
    waste 60s and produce opaque stack traces.
    """
    errors = []

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

    provider = os.getenv("EMBED_PROVIDER", "openai").lower()
    if provider == "voyage":
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
    else:
        if not os.getenv("OPENAI_API_KEY"):
            errors.append(
                "OPENAI_API_KEY no está en el environment (provider default). "
                "Obtén una en https://platform.openai.com/api-keys, "
                "o configura EMBED_PROVIDER=voyage con VOYAGE_API_KEY."
            )
        else:
            try:
                from openai import OpenAI
                OpenAI().embeddings.create(
                    model="text-embedding-3-small",
                    input=["ping"],
                )
            except Exception as exc:
                errors.append(
                    f"OPENAI_API_KEY validación falló: {type(exc).__name__}: {exc}"
                )

    if errors:
        print("✗ TenthAI startup falló:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print("✓ keys validated", file=sys.stderr)


@mcp.tool()
async def decide(question: str, context: str | None = None, skip_scoping: bool = False) -> dict:
    """Mapa de desacuerdo: 9 consejeros + 1 disidente obligatorio (décimo hombre).

    DOS FASES:

    Fase 1 — Scoping (default si no se pasa context):
    Devuelve {status: "needs_context", questions: [...]} con 4-7 preguntas concretas
    tailored al dominio. El caller (Claude Code) debe presentárselas al usuario antes
    de proceder. Sin contexto rico, los consejeros especulan o se quedan genéricos.

    Fase 2 — Run (cuando context está presente o skip_scoping=True):
    Corre 9 consejeros en paralelo + el décimo hombre, devuelve mapa 2D con distancias,
    cita literal del disidente, y abre HTML en navegador.

    Args:
        question: La pregunta de decisión o juicio.
        context: Respuestas del usuario a las preguntas de scoping, o contexto
                 inicial rico. Si está vacío, se entra a fase 1.
        skip_scoping: Si True, salta scoping y corre directo. Útil cuando el caller
                      decide que la pregunta ya tiene contexto suficiente.

    Returns:
        Fase 1: {status: "needs_context", questions: [...], note, next_call_hint}
        Fase 2: {viz_path, frames, tenth_man, summary, cost_clp}
    """
    if not question or not question.strip():
        return {"error": "empty_question", "reason": "Provide una pregunta no vacía."}

    client = AsyncAnthropic()

    if not context and not skip_scoping:
        questions = await generate_questions(client, question)
        if questions is None:
            return {
                "status": "scoping_failed",
                "reason": "No pude generar preguntas de scoping. Pasa skip_scoping=True o provee context para proceder.",
            }
        return {
            "status": "needs_context",
            "questions": questions,
            "note": "Presenta estas preguntas al usuario en un mensaje numerado, espera su respuesta, luego llama decide() de nuevo con question + sus respuestas como context.",
            "next_call_hint": f"decide(question={question!r}, context='<respuestas del usuario formateadas>')",
        }

    try:
        results = await run_agents(client, question, context)
    except RuntimeError as exc:
        return {"error": "agents_failed", "reason": str(exc)}

    successful_frames = [(f, r) for f, r, s in results[:9] if s == "ok"]
    consensus_text = await synthesize_consensus(client, successful_frames, question)

    texts = [r[1] for r in results]
    embed_result = embed_responses(texts)
    if not embed_result["ok"]:
        return embed_result

    proj = project_mds(embed_result["embeddings"])

    cost_clp = 580.0  # rough avg; range CLP 430-730 with 1500/3500 token cap + Haiku consensus + scoping

    html = render(
        question=question,
        results=results,
        consensus=consensus_text,
        coords_2d=proj["coords_2d"],
        distances=proj["distance_to_centroid_of_9"],
        provider=embed_result["provider"],
        model=embed_result["model"],
        cost_estimate_clp=cost_clp,
    )

    # Persist the run: report.html + report.json + regenerate index.html
    distances_list = proj["distance_to_centroid_of_9"]
    coords_list = proj["coords_2d"]
    frame_distances = distances_list[:9]
    max_frame_distance = max(frame_distances)
    min_frame_distance = min(frame_distances)
    most_divergent_frame = results[frame_distances.index(max_frame_distance)][0]
    closest_frame = results[frame_distances.index(min_frame_distance)][0]
    verdict = consensus_verdict(distances_list[9], max_frame_distance)

    report_id = make_report_id(question)
    report_dir = make_report_dir(report_id)
    payload = {
        "schema_version": "1",
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
            for i, (frame, response, status) in enumerate(results[:9])
        ],
        "tenth_man": {
            "response": results[9][1],
            "distance": distances_list[9],
            "embedding_2d": coords_list[9],
        },
        "summary": {
            "tenth_man_distance": distances_list[9],
            "max_frame_distance": max_frame_distance,
            "min_frame_distance": min_frame_distance,
            "most_divergent_frame": most_divergent_frame,
            "closest_frame": closest_frame,
            "consensus_state": verdict["state"],
            "consensus_fragility": verdict["verdict"],
            "n_frames_succeeded": sum(1 for _, _, s in results[:9] if s == "ok"),
        },
        "embed": {
            "provider": embed_result["provider"],
            "model": embed_result["model"],
        },
        "cost_clp": cost_clp,
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

    return {
        "viz_path": viz_path,
        "report_id": report_id,
        "report_dir": str(report_dir),
        "json_path": str(json_path),
        "index_path": str(index_path),
        "viz_note": "Persisted at viz_path (HTML) + json_path (raw data). index_path is the browseable ledger. JSON below contains summaries only (consensus + tenth-man are full).",
        "consensus": consensus_text or "(consensus synthesis failed)",
        "frames": [
            {
                "frame": frame,
                "status": status,
                "distance": round(proj["distance_to_centroid_of_9"][i], 3),
                "summary": _truncate(resp, 300),
            }
            for i, (frame, resp, status) in enumerate(results[:9])
        ],
        "tenth_man": {
            "distance": round(proj["distance_to_centroid_of_9"][9], 3),
            "response": results[9][1],  # full text
        },
        "summary": {
            "tenth_man_distance": round(distances_list[9], 3),
            "max_frame_distance": round(max_frame_distance, 3),
            "consensus_state": verdict["state"],
            "consensus_fragility": verdict["verdict"],
            "n_frames_succeeded": sum(1 for _, _, s in results[:9] if s == "ok"),
            "embed_provider": embed_result["provider"],
            "embed_model": embed_result["model"],
        },
        "cost_clp": cost_clp,
    }


def main():
    _validate_keys_at_startup()
    mcp.run()


if __name__ == "__main__":
    main()
