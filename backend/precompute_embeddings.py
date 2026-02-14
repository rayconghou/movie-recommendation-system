#!/usr/bin/env python3
"""
Standalone script to precompute movie embeddings and store them in the database.
Run from the backend directory with OPENAI_API_KEY set in project root .env:

    cd backend
    python precompute_embeddings.py

After this runs, the FastAPI server will load embeddings from the DB at startup
(no API calls on startup).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Load .env from project root before importing modules that use OPENAI_API_KEY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from database import (
    DEFAULT_DB_PATH,
    ensure_db,
    get_connection,
    load_movies_dataframe,
    save_embeddings_to_dir,
)
from data_loader import enrich_movies_df
from embeddings import (
    OPENAI_EMBEDDING_MODEL,
    is_embeddings_available,
    precompute_embeddings_per_field,
)


def main() -> int:
    if not is_embeddings_available():
        print("OPENAI_API_KEY not set. Set it in project root .env and try again.", file=sys.stderr)
        return 1

    movies_csv = PROJECT_ROOT / "movies_metadata.csv"
    if not movies_csv.exists():
        print(f"movies_metadata.csv not found at {movies_csv}", file=sys.stderr)
        return 1

    print("Loading movies and enriching with credits/keywords...")
    conn = ensure_db(movies_csv, DEFAULT_DB_PATH)
    df = load_movies_dataframe(conn)
    conn.close()
    df = enrich_movies_df(df, PROJECT_ROOT)
    df["combined_text"] = df.get("combined_text", "").fillna("").astype(str)

    embed_fields = ["title", "overview", "genres", "actors", "keywords"]
    print(f"Precomputing embeddings for {len(df)} movies, fields: {embed_fields}")
    result = precompute_embeddings_per_field(df, embed_fields, model=OPENAI_EMBEDDING_MODEL)
    if result is None:
        print("Precompute failed (see errors above).", file=sys.stderr)
        return 1

    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    print("Saving embeddings to", embeddings_dir, "...")
    save_embeddings_to_dir(embeddings_dir, result)
    for field, matrix in result.items():
        print(f"  Saved {field}: shape {matrix.shape}")
    print("Done. Restart the backend to use precomputed embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
