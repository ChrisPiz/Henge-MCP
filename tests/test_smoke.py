"""Smoke tests — basic startup invariants + error path coverage."""
from pathlib import Path
import pytest


def test_prompts_loaded_at_startup():
    """Los 10 prompts cargan al import. Ninguno vacío. Single source of truth."""
    from henge.agents import PROMPTS

    expected_keys = {
        "empirical", "historical", "first-principles", "analogical",
        "systemic", "ethical", "soft-contrarian", "radical-optimist",
        "pre-mortem", "tenth-man",
    }
    assert set(PROMPTS.keys()) == expected_keys
    for name, text in PROMPTS.items():
        assert text and len(text) > 50, f"Prompt {name} too short or empty"


def test_html_renders(synthetic_embeddings_10):
    """render() returns a valid HTML string with all 10 points + section markers.

    render() is now pure — it returns a string instead of writing to disk.
    Persistence + browser-open live in server.py via storage.write_record.
    """
    from henge.embed import project_mds
    from henge import viz

    proj = project_mds(synthetic_embeddings_10)

    results = [
        (f"frame{i}", f"respuesta {i}", "ok") for i in range(9)
    ] + [("tenth-man", "respuesta de disenso", "ok")]

    html = viz.render(
        question="¿Deberíamos lanzar ahora?",
        results=results,
        coords_2d=proj["coords_2d"],
        distances=proj["distance_to_centroid_of_9"],
        provider="openai",
        model="text-embedding-3-small",
        cost_estimate_clp=350,
    )

    assert isinstance(html, str) and len(html) > 1000
    lower = html.lower()
    assert "<html" in lower
    assert "henge" in lower
    assert "tenth-man" in lower
    # All 9 frame names should be present
    for i in range(9):
        assert f"frame{i}" in html


def test_storage_persists_and_indexes(tmp_path, monkeypatch):
    """write_record + write_index produce a browseable ledger of past runs."""
    monkeypatch.setenv("HENGE_REPORTS_DIR", str(tmp_path))
    # Re-import storage so REPORTS_DIR picks up the patched env var.
    import importlib
    from henge import storage as storage_module
    storage = importlib.reload(storage_module)

    rid = storage.make_report_id("Should I ship this Friday?")
    rdir = storage.make_report_dir(rid)
    payload = {
        "schema_version": "1",
        "id": rid,
        "timestamp": "2026-04-30T12:00:00-04:00",
        "question": "Should I ship this Friday?",
        "summary": {
            "tenth_man_distance": 0.143,
            "consensus_state": "aligned-stable",
            "consensus_fragility": "Consejeros alineados — el disenso suena pero el consenso aguanta.",
        },
    }
    html_path, json_path = storage.write_record(rdir, "<html>x</html>", payload)
    index_path = storage.write_index()

    assert html_path.exists() and json_path.exists() and index_path.exists()
    records = storage.list_records()
    assert len(records) == 1
    assert records[0]["question"] == "Should I ship this Friday?"
    index_html = index_path.read_text(encoding="utf-8")
    assert "Should I ship this Friday?" in index_html
    assert "open ↗" in index_html
    # New aligned-stable state surfaces as "aligned" in the index column
    assert ">aligned<" in index_html


def test_consensus_verdict_three_states():
    """Verdict picks the right state for tight/fragile/divided shapes."""
    from henge.viz import consensus_verdict

    # 9 tight (max 0.08), tenth moderate (0.10) → aligned-stable
    v = consensus_verdict(tenth_distance=0.10, max_frame_distance=0.08)
    assert v["state"] == "aligned-stable"
    assert "alineados" in v["verdict"].lower()

    # 9 tight (max 0.119), tenth pushed (0.213) → aligned-fragile
    v = consensus_verdict(tenth_distance=0.213, max_frame_distance=0.119)
    assert v["state"] == "aligned-fragile"
    assert "frágil" in v["verdict"].lower()

    # 9 spread (max 0.22) → divided regardless of tenth
    v = consensus_verdict(tenth_distance=0.40, max_frame_distance=0.22)
    assert v["state"] == "divided"
    assert "divididos" in v["verdict"].lower()


def test_voyage_failure_returns_structured_error(monkeypatch):
    """Si embed provider falla, embed_responses retorna {ok: False, error, reason}.

    Sin esto, MCP server propaga stack trace cruda → confusión del developer.
    """
    from henge import embed

    def boom(*args, **kwargs):
        raise RuntimeError("Simulated voyage 500")

    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    monkeypatch.setattr(embed, "_embed_voyage", boom)
    # Ensure cache misses so embed_fn actually runs
    monkeypatch.setattr(embed, "_cached_embedding", lambda *a, **kw: None)
    monkeypatch.setattr(embed, "_save_embedding", lambda *a, **kw: None)

    result = embed.embed_responses(["text1", "text2"])

    assert result["ok"] is False
    assert result["error"] == "embed_failed"
    assert "voyage" in result["reason"].lower() or "500" in result["reason"]


def test_startup_validates_keys_missing(monkeypatch, capsys):
    """If ANTHROPIC_API_KEY is missing, startup exits with a clear stderr message.

    NOTE: server.py calls load_dotenv() at module import. To exercise the
    "missing key" branch we must clear the env vars *after* the import has
    populated them, otherwise the validator sees the keys from .env.
    """
    # Import first so load_dotenv has already run and pollued os.environ.
    from henge.server import _validate_keys_at_startup

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)

    with pytest.raises(SystemExit):
        _validate_keys_at_startup()

    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err
