"""
Embedding-based retrieval: OpenAI text-embedding-3-small, precomputed per-field movie embeddings,
query embedding and weighted cosine similarity over candidates.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Max batch size for OpenAI embeddings API (smaller to avoid token limits on long overviews)
EMBED_BATCH_SIZE = 50
# Max chars per text to stay under model context (e.g. 8191 tokens ~ 32k chars total per batch)
MAX_TEXT_LEN = 2000


def _get_client():
    if not OPENAI_API_KEY.strip():
        return None
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: List[str], model: str = OPENAI_EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """
    Embed a list of texts with OpenAI. Returns (n, dim) float32 array or None on failure.
    """
    client = _get_client()
    if client is None:
        return None
    if not texts:
        return None
    cleaned = [t if t.strip() else " " for t in texts]
    try:
        out = client.embeddings.create(input=cleaned, model=model)
        items = sorted(out.data, key=lambda x: getattr(x, "index", 0))
        return np.array([item.embedding for item in items], dtype=np.float32)
    except Exception:
        return None


def _truncate_for_embedding(text: str) -> str:
    """Truncate long text to avoid token limit errors."""
    if len(text) <= MAX_TEXT_LEN:
        return text
    return text[: MAX_TEXT_LEN - 1].rsplit(" ", 1)[0] if " " in text[:MAX_TEXT_LEN] else text[:MAX_TEXT_LEN]


def embed_texts_batched(
    texts: List[str],
    model: str = OPENAI_EMBEDDING_MODEL,
    show_progress: bool = True,
) -> Optional[np.ndarray]:
    """Embed texts in batches; returns (n, dim) or None."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    client = _get_client()
    if client is None or not texts:
        return None
    cleaned = [_truncate_for_embedding(t if t.strip() else " ") for t in texts]
    all_embeddings: List[List[float]] = []
    batch_starts = list(range(0, len(cleaned), EMBED_BATCH_SIZE))
    num_batches = len(batch_starts)
    iterator = (
        tqdm(batch_starts, desc="Embedding", unit="batch", total=num_batches)
        if (show_progress and tqdm is not None)
        else batch_starts
    )
    for i in iterator:
        batch = cleaned[i : i + EMBED_BATCH_SIZE]
        try:
            out = client.embeddings.create(input=batch, model=model)
            # Preserve order by index (Embedding has .index)
            items = sorted(out.data, key=lambda x: getattr(x, "index", 0))
            for item in items:
                all_embeddings.append(item.embedding)
        except Exception as e:
            print("[embeddings] Batch embed failed:", e, file=sys.stderr)
            return None
    if len(all_embeddings) != len(texts):
        print("[embeddings] Length mismatch after embed", len(all_embeddings), "vs", len(texts), file=sys.stderr)
        return None
    return np.array(all_embeddings, dtype=np.float32)


def build_field_texts(df: pd.DataFrame, field: str) -> List[str]:
    """
    Build one string per row for the given field.
    Fields: title, overview, genres, actors, keywords.
    """
    n = len(df)
    if field == "title":
        return [str(row.get("title", "") or "").strip() for _, row in df.iterrows()]
    if field == "overview":
        return [str(row.get("overview", "") or "").strip() for _, row in df.iterrows()]
    if field == "genres":
        return [
            " ".join(row.get("parsed_genres") or [])
            for _, row in df.iterrows()
        ]
    if field == "actors":
        if "actors" not in df.columns:
            return [" "] * n
        return [
            " ".join(row.get("actors") or [])
            for _, row in df.iterrows()
        ]
    if field == "keywords":
        if "keywords" not in df.columns:
            return [" "] * n
        return [
            " ".join(row.get("keywords") or [])
            for _, row in df.iterrows()
        ]
    return [" "] * n


def precompute_embeddings_per_field(
    df: pd.DataFrame,
    fields: List[str],
    model: str = OPENAI_EMBEDDING_MODEL,
) -> Optional[Dict[str, np.ndarray]]:
    """
    For each field in fields, build texts and embed; return dict field -> (n_movies, dim).
    Returns None if embeddings unavailable (no API key or error).
    """
    if not _get_client():
        return None
    result: Dict[str, np.ndarray] = {}
    n = len(df)
    for field in fields:
        print(f"[embeddings] Precomputing field '{field}' ({n} texts)...", flush=True)
        texts = build_field_texts(df, field)
        mat = embed_texts_batched(texts, model=model)
        if mat is None:
            print(f"[embeddings] Precompute failed at field '{field}'", file=sys.stderr)
            return None
        result[field] = mat
    return result


def build_query_text_for_field(intent: object, field: str) -> str:
    """
    Build the query-side text used to embed for a given field.
    intent must have: free_text, keywords, genres, featured_actors.
    """
    if field == "title":
        return intent.free_text
    if field == "overview":
        parts = [intent.free_text] + intent.keywords
        return " ".join(p for p in parts if p).strip() or " "
    if field == "genres":
        return " ".join(intent.genres) or " "
    if field == "actors":
        return " ".join(intent.featured_actors) or " "
    if field == "keywords":
        return " ".join(intent.keywords) or " "
    return intent.free_text or " "


def query_embedding_for_fields(
    intent: object,
    fields: List[str],
    model: str = OPENAI_EMBEDDING_MODEL,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Embed the query text for each field. Returns dict field -> (1, dim) vector.
    """
    client = _get_client()
    if client is None:
        return None
    result: Dict[str, np.ndarray] = {}
    for field in fields:
        text = build_query_text_for_field(intent, field)
        vec = embed_texts([text], model=model)
        if vec is None:
            return None
        result[field] = vec
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n_a, dim), b: (n_b, dim). Returns (n_a, n_b) or (n_a,) if b is (1, dim)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)


def weighted_similarity_over_candidates(
    candidate_indices: np.ndarray,
    field_embeddings: Dict[str, np.ndarray],
    query_embeddings: Dict[str, np.ndarray],
    fields_to_use: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    For each field in fields_to_use, compute cosine similarity between query embedding (1, dim)
    and candidate movie embeddings (len(candidate_indices), dim). Combine with optional weights.
    Returns 1d array of length len(candidate_indices).
    """
    if not fields_to_use:
        return np.zeros(len(candidate_indices), dtype=np.float32)
    if weights is None:
        weights = {f: 1.0 for f in fields_to_use}
    total_weight = sum(weights.get(f, 1.0) for f in fields_to_use) or 1.0
    combined = np.zeros(len(candidate_indices), dtype=np.float32)
    for field in fields_to_use:
        if field not in field_embeddings or field not in query_embeddings:
            continue
        movie_mat = field_embeddings[field]  # (n_movies, dim)
        candidate_mat = movie_mat[candidate_indices]  # (n_candidates, dim)
        q_vec = query_embeddings[field]  # (1, dim)
        sim = cosine_similarity(q_vec, candidate_mat)
        sim_1d = sim.flatten()
        w = weights.get(field, 1.0) / total_weight
        combined += w * sim_1d
    return combined


def is_embeddings_available() -> bool:
    return bool(OPENAI_API_KEY.strip())
