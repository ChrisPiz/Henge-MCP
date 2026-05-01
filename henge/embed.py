"""Embeddings (OpenAI default, Voyage opt-in) + classical MDS over pairwise cosine distance.

Why MDS not PCA: with N=10 in 1024+ dims (n<<d), PCA is statistically trivial —
first 2 components capture ~100% variance regardless of semantic content.
MDS preserves pairwise distances faithfully, which IS what a disagreement map needs.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

CACHE_DIR = Path(".embed_cache")


def _cache_key(text: str, provider: str, model: str) -> str:
    return hashlib.sha256(f"{provider}:{model}:{text}".encode()).hexdigest()


def _cached_embedding(text: str, provider: str, model: str):
    if not CACHE_DIR.exists():
        return None
    path = CACHE_DIR / f"{_cache_key(text, provider, model)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_embedding(text: str, provider: str, model: str, embedding):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(text, provider, model)}.json"
    path.write_text(json.dumps(embedding))


def _embed_openai(texts, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _embed_voyage(texts, model="voyage-3-large"):
    import voyageai
    client = voyageai.Client()
    resp = client.embed(texts=texts, model=model)
    return resp.embeddings


def _resolve_provider():
    """Return (provider, model, embed_fn). Default OpenAI for lower friction."""
    provider = os.getenv("EMBED_PROVIDER", "openai").lower()
    if provider == "voyage":
        return "voyage", "voyage-3-large", _embed_voyage
    return "openai", "text-embedding-3-small", _embed_openai


def embed_responses(texts):
    """Batch embed N texts. Returns dict {ok, embeddings?, provider, model, error?, reason?}.

    Errors propagate as structured dict, not raw exception, so MCP server returns
    a clear error to the client instead of a stack trace.
    """
    provider, model, embed_fn = _resolve_provider()

    embeddings = [_cached_embedding(t, provider, model) for t in texts]
    missing_idx = [i for i, e in enumerate(embeddings) if e is None]

    if missing_idx:
        try:
            new_embeds = embed_fn([texts[i] for i in missing_idx])
            for i, e in zip(missing_idx, new_embeds):
                embeddings[i] = e
                try:
                    _save_embedding(texts[i], provider, model, e)
                except Exception:
                    pass  # cache write is best-effort
        except Exception as exc:
            return {
                "ok": False,
                "error": "embed_failed",
                "reason": f"{provider}: {type(exc).__name__}: {exc}",
            }

    return {
        "ok": True,
        "embeddings": embeddings,
        "provider": provider,
        "model": model,
    }


def project_mds(embeddings):
    """Classical MDS over pairwise cosine distance → 2D coords.

    centroid_of_9 is computed only over the first 9 embeddings (frames),
    excluding index 9 (the tenth-man). Distances are computed in the original
    embedding space using cosine distance — the MDS projection is for viz only.
    """
    arr = np.array(embeddings, dtype=float)
    if arr.shape[0] != 10:
        raise ValueError(f"Expected 10 embeddings, got {arr.shape[0]}")

    cosine_distances = squareform(pdist(arr, metric="cosine"))

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=4,
    )
    coords_2d = mds.fit_transform(cosine_distances)

    centroid_of_9 = arr[:9].mean(axis=0)
    centroid_norm = centroid_of_9 / (np.linalg.norm(centroid_of_9) + 1e-12)

    distances = []
    for vec in arr:
        v_norm = vec / (np.linalg.norm(vec) + 1e-12)
        cos_sim = float(np.dot(v_norm, centroid_norm))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        distances.append(1.0 - cos_sim)

    return {
        "coords_2d": coords_2d.tolist(),
        "distance_to_centroid_of_9": distances,
    }
