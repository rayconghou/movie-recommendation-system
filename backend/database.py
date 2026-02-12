"""
SQLite database for deterministic movie filtering (e.g. by release decade).
Populated from movies_metadata.csv; used to get candidate row indices by year range.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "movies.db"


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


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS movies (
            row_index INTEGER PRIMARY KEY,
            tmdb_id INTEGER,
            title TEXT NOT NULL,
            overview TEXT,
            genres_raw TEXT,
            poster_path TEXT,
            release_date TEXT,
            release_year INTEGER,
            combined_text TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_movies_release_year ON movies(release_year)"
    )
    conn.commit()


def is_populated(conn: sqlite3.Connection) -> bool:
    cur = conn.execute("SELECT COUNT(*) FROM movies")
    return cur.fetchone()[0] > 0


def populate_from_csv(conn: sqlite3.Connection, csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df["parsed_genres"] = df["genres"].apply(_safe_parse_genres)
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

    conn.execute("DELETE FROM movies")
    for i, row in df.iterrows():
        release_year = None
        if pd.notna(row.get("release_year")):
            try:
                release_year = int(row["release_year"])
            except (ValueError, TypeError):
                pass
        conn.execute(
            """
            INSERT INTO movies (
                row_index, tmdb_id, title, overview, genres_raw, poster_path,
                release_date, release_year, combined_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                i,
                row.get("id"),
                str(row.get("title", "")),
                str(row.get("overview", "")) if pd.notna(row.get("overview")) else None,
                str(row.get("genres", "")) if pd.notna(row.get("genres")) else None,
                str(row.get("poster_path", "")) if pd.notna(row.get("poster_path")) else None,
                str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else None,
                release_year,
                str(row.get("combined_text", "")),
            ),
        )
    conn.commit()


def ensure_db(csv_path: Path, db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create DB and schema; populate from CSV if table is empty. Returns connection."""
    conn = get_connection(db_path)
    init_schema(conn)
    if not is_populated(conn):
        populate_from_csv(conn, csv_path)
    return conn


def get_candidate_row_indices(
    conn: sqlite3.Connection,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Optional[List[int]]:
    """
    Return list of row_index values for movies in the given year range (inclusive).
    If both start_year and end_year are None, returns None (meaning no filter / use all rows).
    """
    if start_year is None and end_year is None:
        return None
    query = "SELECT row_index FROM movies WHERE 1=1"
    params: List[object] = []
    if start_year is not None:
        query += " AND release_year >= ?"
        params.append(start_year)
    if end_year is not None:
        query += " AND release_year <= ?"
        params.append(end_year)
    query += " ORDER BY row_index"
    rows = conn.execute(query, params).fetchall()
    return [r[0] for r in rows]


def load_movies_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load full movies table into a DataFrame with columns matching Recommender's movies_df."""
    df = pd.read_sql_query(
        """
        SELECT row_index, tmdb_id AS id, title, overview, genres_raw AS genres,
               poster_path, release_date, release_year, combined_text
        FROM movies ORDER BY row_index
        """,
        conn,
    )
    df["parsed_genres"] = df["genres"].apply(_safe_parse_genres)
    df["overview"] = df["overview"].fillna("")
    return df
