from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import time

# Load .env from project root so OPENAI_API_KEY is available to embeddings/query_parser
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from database import (
    DEFAULT_DB_PATH,
    ensure_db,
    get_candidate_row_indices,
    get_connection,
    load_embeddings_from_dir,
    load_embeddings_from_db,
    load_movies_dataframe,
)
from data_loader import enrich_movies_df
from embeddings import (
    query_embedding_for_fields,
    weighted_similarity_over_candidates,
)
from query_parser import parse_query_for_search


MOVIES_CSV_PATH = PROJECT_ROOT / "movies_metadata.csv"


def parse_year_range_deterministic(query: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse release year range from natural language using deterministic rules (no LLM).
    E.g. "90s", "1990s" -> (1990, 1999); "early 2000s" -> (2000, 2004).
    Returns (start_year, end_year) inclusive; (None, None) if no range detected.
    """
    q = " " + query.lower().strip() + " "
    if not q.strip():
        return None, None

    def decade_to_range(decade_phrase: str, base_year: int) -> Tuple[Optional[int], Optional[int]]:
        # Check for "early" or "late" immediately before the decade phrase
        idx = q.find(decade_phrase)
        if idx > 0:
            before = q[:idx].rstrip()
            if before.endswith("late"):
                return base_year + 5, base_year + 9
            if before.endswith("early"):
                return base_year, base_year + 4
        return base_year, base_year + 9

    # "90s", "80s", "70s", "00s" (two-digit decade)
    m = re.search(r"\b(\d)0s\b", q)
    if m:
        d = int(m.group(1))
        if d >= 8 and d <= 9:
            base = 1900 + d * 10
        elif d == 0:
            base = 2000
        else:
            base = 1900 + d * 10
        return decade_to_range(m.group(0), base)

    # "1990s", "1980s", "2000s"
    m = re.search(r"\b(19\d{2})s\b", q)
    if m:
        base = int(m.group(1))
        return decade_to_range(m.group(0), base)
    m = re.search(r"\b(20\d{2})s\b", q)
    if m:
        base = int(m.group(1))
        return decade_to_range(m.group(0), base)

    return None, None


class RecommendRequest(BaseModel):
    query: str


class MovieResponse(BaseModel):
    rank: int
    title: str
    poster_path: Optional[str]
    overview_truncated: Optional[str]
    genres: List[str]
    release_year: Optional[int] | None = None
    score: float
    justification: str


class Recommender:
    def __init__(self, csv_path: Path, db_path: Optional[Path] = None) -> None:
        self.csv_path = csv_path
        self.db_path: Optional[Path] = db_path
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: TfidfVectorizer
        self.tfidf_matrix = None
        self.feature_names: np.ndarray | None = None

        # Optional ANN index and diagnostics
        self.use_ann_index: bool = False
        self.nn_index: NearestNeighbors | None = None
        self.enable_timing: bool = False
        # Precomputed per-field embeddings (field -> (n_movies, dim)); None if not available
        self.field_embeddings: Optional[Dict[str, np.ndarray]] = None

        self._load_and_fit()

    @staticmethod
    def _safe_parse_genres(genres_raw: str) -> List[str]:
        if not isinstance(genres_raw, str) or not genres_raw.strip():
            return []
        try:
            data = ast.literal_eval(genres_raw)
            if isinstance(data, list):
                names = [d.get("name") for d in data if isinstance(d, dict) and d.get("name")]
                return [str(name) for name in names]
        except (SyntaxError, ValueError):
            return []
        return []

    def _load_and_fit(self) -> None:
        if self.db_path is not None:
            ensure_db(self.csv_path, self.db_path)
            conn = get_connection(self.db_path)
            df = load_movies_dataframe(conn)
            conn.close()
            df["combined_text"] = df["combined_text"].fillna("").astype(str)
            # Rows are ORDER BY row_index, so df row i = row_index i for filtering
        else:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"movies_metadata.csv not found at {self.csv_path}")
            df = pd.read_csv(self.csv_path, low_memory=False)
            df = df.dropna(subset=["title"]).reset_index(drop=True)
            df["parsed_genres"] = df["genres"].apply(self._safe_parse_genres)
            df["overview"] = df["overview"].fillna("")
            if "release_date" in df.columns:
                release_dt = pd.to_datetime(df["release_date"], errors="coerce")
                df["release_year"] = release_dt.dt.year
            else:
                df["release_year"] = pd.NA
            def build_text(row: pd.Series) -> str:
                title = str(row.get("title", "") or "")
                overview = str(row.get("overview", "") or "")
                genres = " ".join(row.get("parsed_genres", []) or [])
                return " ".join([part for part in [title, overview, genres] if part])
            df["combined_text"] = df.apply(build_text, axis=1)

        # Enrich with credits (actors) and keywords; join at load time by id
        df = enrich_movies_df(df, PROJECT_ROOT)

        self.movies_df = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies_df["combined_text"].fillna("").astype(str)
        )
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Load precomputed embeddings from data/embeddings/*.npy (from precompute_embeddings.py); no API calls at startup
        embed_fields = ["title", "overview", "genres", "actors", "keywords"]
        if self.db_path is not None:
            embeddings_dir = self.db_path.parent / "embeddings"
            self.field_embeddings = load_embeddings_from_dir(embeddings_dir, embed_fields)
            if self.field_embeddings is None:
                conn = get_connection(self.db_path)
                try:
                    self.field_embeddings = load_embeddings_from_db(conn, embed_fields)
                finally:
                    conn.close()
            if self.field_embeddings is not None:
                print("[Recommender] Loaded embeddings for fields:", list(self.field_embeddings.keys()))
            else:
                print("[Recommender] No precomputed embeddings. Run: cd backend && python precompute_embeddings.py")
        else:
            self.field_embeddings = None
            print("[Recommender] No DB path; embedding search disabled. Use DB + precompute_embeddings.py for embeddings.")

        # Optionally build an approximate nearest-neighbor style index over the TF-IDF matrix.
        # This can be enabled by setting `self.use_ann_index = True` after initialization.
        if self.use_ann_index:
            nn = NearestNeighbors(metric="cosine", algorithm="brute")
            nn.fit(self.tfidf_matrix)
            self.nn_index = nn

    def _truncate(self, text: Optional[str], max_chars: int = 220) -> Optional[str]:
        if text is None:
            return None
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"

    def _build_justification(
        self,
        query: str,
        query_vec,
        movie_index: int,
        top_k_terms: int = 3,
    ) -> str:
        if self.tfidf_matrix is None or self.feature_names is None:
            return "Ranked based on overall similarity to your query."

        movie_vec = self.tfidf_matrix[movie_index]
        shared = movie_vec.multiply(query_vec)
        if shared.nnz > 0:
            shared_coo = shared.tocoo()
            idx_sorted = np.argsort(shared_coo.data)[::-1]
            top_indices = shared_coo.col[idx_sorted][:top_k_terms]
            terms = [self.feature_names[i] for i in top_indices]
        else:
            terms = []

        query_tokens = {t.lower() for t in query.split()}
        genres = self.movies_df.iloc[movie_index]["parsed_genres"] or []
        matched_genres = [g for g in genres if g.lower() in query_tokens]

        parts: List[str] = []
        if terms:
            parts.append("shares themes like " + ", ".join(terms))
        if matched_genres:
            parts.append("matches genres such as " + ", ".join(matched_genres))

        if parts:
            return "Ranked highly because it " + " and ".join(parts) + "."

        return "Ranked based on textual similarity between your query and the movie's title, description, and genres."

    def recommend(self, query: str, top_k: int = 30) -> List[MovieResponse]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF matrix is not initialized.")

        # LLM query parser: structured intent (genres, keywords, actors, years, fields_to_use).
        intent = parse_query_for_search(query)
        start_year, end_year = intent.start_year, intent.end_year
        if start_year is None and end_year is None:
            start_year, end_year = parse_year_range_deterministic(query)

        # Build candidate index set: use DB when available, else filter on dataframe.
        df = self.movies_df
        all_indices = np.arange(df.shape[0])
        if self.db_path is not None:
            conn = get_connection(self.db_path)
            try:
                candidate_list = get_candidate_row_indices(conn, start_year, end_year)
                if candidate_list is not None:
                    candidate_indices = np.array(candidate_list, dtype=np.intp)
                else:
                    candidate_indices = all_indices
            finally:
                conn.close()
        else:
            if start_year is None and end_year is None:
                candidate_indices = all_indices
            else:
                years = df.get("release_year")
                if years is not None:
                    mask = years.notna()
                    if start_year is not None:
                        mask &= years >= start_year
                    if end_year is not None:
                        mask &= years <= end_year
                    candidate_indices = np.flatnonzero(mask.to_numpy())
                else:
                    candidate_indices = all_indices

        # If filtering yields no candidates, fall back to the full catalog.
        if candidate_indices.size == 0:
            candidate_indices = all_indices

        num_candidates = candidate_indices.size
        top_k = min(max(top_k, 0), num_candidates)

        if top_k == 0:
            return []

        t0 = time.perf_counter()
        use_embedding = (
            self.field_embeddings is not None
            and intent.fields_to_use
            and all(
                f in self.field_embeddings for f in intent.fields_to_use
            )
        )
        query_embeddings = None
        if use_embedding:
            query_embeddings = query_embedding_for_fields(
                intent, intent.fields_to_use
            )
            if query_embeddings is None:
                use_embedding = False

        if use_embedding and query_embeddings is not None:
            print("[Recommender] Using embedding-based similarity (fields:", intent.fields_to_use, ")")
            scores = weighted_similarity_over_candidates(
                candidate_indices,
                self.field_embeddings,
                query_embeddings,
                intent.fields_to_use,
            )
            if top_k >= len(scores):
                top_local_indices = np.argsort(scores)[::-1]
            else:
                partition_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_local_indices = partition_indices[
                    np.argsort(scores[partition_indices])[::-1]
                ]
            cosine_similarities = scores
        else:
            print("[Recommender] Using TF-IDF similarity (embedding path unavailable or failed)")
            query_vec = self.vectorizer.transform([query])
            use_ann = self.use_ann_index and self.nn_index is not None and (
                start_year is None and end_year is None
            )
            if use_ann:
                distances, indices = self.nn_index.kneighbors(
                    query_vec, n_neighbors=top_k
                )
                top_local_indices = indices[0]
                cosine_similarities = (1.0 - distances[0]).astype(np.float64)
            else:
                sub_matrix = self.tfidf_matrix[candidate_indices]
                cosine_similarities = linear_kernel(
                    query_vec, sub_matrix
                ).flatten().astype(np.float64)
                if top_k >= cosine_similarities.size:
                    top_local_indices = np.argsort(cosine_similarities)[::-1]
                else:
                    partition_indices = np.argpartition(
                        cosine_similarities, -top_k
                    )[-top_k:]
                    top_local_indices = partition_indices[
                        np.argsort(cosine_similarities[partition_indices])[::-1]
                    ]

        t2 = time.perf_counter()
        t1 = t0  # for timing message

        movies: List[MovieResponse] = []
        detailed_justification_k = 10
        year_filter_active = start_year is not None or end_year is not None
        query_vec = None
        if not use_embedding:
            query_vec = self.vectorizer.transform([query])
        for rank_idx, local_idx in enumerate(top_local_indices, start=1):
            movie_idx = candidate_indices[local_idx]
            row = self.movies_df.iloc[movie_idx]
            title = str(row.get("title", ""))
            year_value = row.get("release_year")
            release_year = int(year_value) if pd.notna(year_value) else None
            poster_path = row.get("poster_path")
            poster_path = str(poster_path) if isinstance(poster_path, str) and poster_path.strip() else None
            overview = row.get("overview")
            overview_truncated = self._truncate(str(overview) if isinstance(overview, str) else None)
            genres = row.get("parsed_genres") or []
            score = float(cosine_similarities[local_idx])
            if rank_idx <= detailed_justification_k and query_vec is not None:
                justification = self._build_justification(query, query_vec, movie_idx)
            else:
                justification = (
                    "Ranked based on similarity to your query (embedding or text match)."
                )

            if year_filter_active:
                if start_year is not None and end_year is not None:
                    justification += f" Released between {start_year} and {end_year}."
                elif start_year is not None:
                    justification += f" Released in or after {start_year}."
                elif end_year is not None:
                    justification += f" Released in or before {end_year}."

            movies.append(
                MovieResponse(
                    rank=rank_idx,
                    title=title,
                    poster_path=poster_path,
                    overview_truncated=overview_truncated,
                    genres=genres,
                    release_year=release_year,
                    score=score,
                    justification=justification,
                )
            )

        t3 = time.perf_counter()
        if self.enable_timing:
            print(
                f"[Recommender] timings year_range=({start_year},{end_year}) "
                f"candidates={num_candidates}/{self.movies_df.shape[0]} "
                f"query_vec={t1 - t0:.4f}s similarity={t2 - t1:.4f}s "
                f"responses={t3 - t2:.4f}s total={t3 - t0:.4f}s"
            )

        return movies


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender(MOVIES_CSV_PATH, db_path=DEFAULT_DB_PATH)


app = FastAPI(title="Movie Recommendation API")


@app.post("/recommend", response_model=List[MovieResponse])
def recommend_movies(payload: RecommendRequest) -> List[MovieResponse]:
    query = payload.query or ""
    try:
        recommender = get_recommender()
        results = recommender.recommend(query=query, top_k=30)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return results


@app.on_event("startup")
def startup_event() -> None:
    """
    Warm up the recommender on server startup so the first user request
    does not pay the cost of loading the CSV and fitting TF-IDF.
    """
    _ = get_recommender()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

