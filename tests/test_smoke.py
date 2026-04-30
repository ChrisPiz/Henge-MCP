"""Smoke tests — basic startup invariants + error path coverage."""
from pathlib import Path
import pytest


def test_prompts_loaded_at_startup():
    """Los 10 prompts cargan al import. Ninguno vacío. Single source of truth."""
    from tenthai.agents import PROMPTS

    expected_keys = {
        "empirical", "historical", "first-principles", "analogical",
        "systemic", "ethical", "soft-contrarian", "radical-optimist",
        "pre-mortem", "tenth-man",
    }
    assert set(PROMPTS.keys()) == expected_keys
    for name, text in PROMPTS.items():
        assert text and len(text) > 50, f"Prompt {name} too short or empty"


def test_html_renders(tmp_path, monkeypatch, synthetic_embeddings_10):
    """render() produce un archivo HTML válido con los 10 puntos en el scatter."""
    from tenthai.embed import project_mds
    from tenthai import viz

    monkeypatch.setattr(viz.webbrowser, "open", lambda *a, **kw: None)
    monkeypatch.setattr(viz.tempfile, "gettempdir", lambda: str(tmp_path))

    proj = project_mds(synthetic_embeddings_10)

    results = [
        (f"frame{i}", f"respuesta {i}", "ok") for i in range(9)
    ] + [("tenth-man", "respuesta de disenso", "ok")]

    path = viz.render(
        question="¿Deberíamos lanzar ahora?",
        results=results,
        coords_2d=proj["coords_2d"],
        distances=proj["distance_to_centroid_of_9"],
        provider="openai",
        model="text-embedding-3-small",
        cost_estimate_clp=350,
    )

    assert Path(path).exists()
    html = Path(path).read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "tenthai" in html.lower()


def test_voyage_failure_returns_structured_error(monkeypatch):
    """Si embed provider falla, embed_responses retorna {ok: False, error, reason}.

    Sin esto, MCP server propaga stack trace cruda → confusión del developer.
    """
    from tenthai import embed

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
    """Si ANTHROPIC_API_KEY falta, startup sale con mensaje claro y SystemExit."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)

    from tenthai.server import _validate_keys_at_startup

    with pytest.raises(SystemExit):
        _validate_keys_at_startup()

    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.err
