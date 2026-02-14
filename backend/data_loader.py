"""
Load credits.csv and keywords.csv and enrich movies DataFrame with actors and keywords.
Join by TMDB id (id in metadata/credits/keywords).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pandas as pd


def _safe_parse_cast(cast_raw: str) -> List[str]:
    """Extract actor names from credits cast column (list of dicts with 'name')."""
    if not isinstance(cast_raw, str) or not cast_raw.strip():
        return []
    try:
        data = ast.literal_eval(cast_raw)
        if isinstance(data, list):
            return [str(d.get("name", "")) for d in data if isinstance(d, dict) and d.get("name")]
    except (SyntaxError, ValueError):
        return []
    return []


def _safe_parse_keywords(keywords_raw: str) -> List[str]:
    """Extract keyword names from keywords column (list of dicts with 'name')."""
    if not isinstance(keywords_raw, str) or not keywords_raw.strip():
        return []
    try:
        data = ast.literal_eval(keywords_raw)
        if isinstance(data, list):
            return [str(d.get("name", "")) for d in data if isinstance(d, dict) and d.get("name")]
    except (SyntaxError, ValueError):
        return []
    return []


def load_credits(credits_path: Path) -> pd.DataFrame:
    """Load credits.csv and return a DataFrame with id and actors (list of names)."""
    if not credits_path.exists():
        return pd.DataFrame(columns=["id", "actors"])
    df = pd.read_csv(credits_path, low_memory=False)
    if "cast" not in df.columns or "id" not in df.columns:
        return pd.DataFrame(columns=["id", "actors"])
    df["actors"] = df["cast"].apply(_safe_parse_cast)
    return df[["id", "actors"]].copy()


def load_keywords(keywords_path: Path) -> pd.DataFrame:
    """Load keywords.csv and return a DataFrame with id and keywords (list of names)."""
    if not keywords_path.exists():
        return pd.DataFrame(columns=["id", "keywords"])
    df = pd.read_csv(keywords_path, low_memory=False)
    if "keywords" not in df.columns or "id" not in df.columns:
        return pd.DataFrame(columns=["id", "keywords"])
    df["keywords"] = df["keywords"].apply(_safe_parse_keywords)
    return df[["id", "keywords"]].copy()


def enrich_movies_df(movies_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """
    Left-join credits and keywords onto movies_df by id.
    Expects movies_df to have column 'id' (tmdb_id). Adds 'actors' and 'keywords' (lists of str).
    """
    df = movies_df.copy()
    id_col = "id"
    if id_col not in df.columns and "tmdb_id" in df.columns:
        df = df.rename(columns={"tmdb_id": id_col})
    if id_col not in df.columns:
        df["actors"] = [[]] * len(df)
        df["keywords"] = [[]] * len(df)
        return df

    credits_path = project_root / "credits.csv"
    keywords_path = project_root / "keywords.csv"

    credits = load_credits(credits_path)
    keywords = load_keywords(keywords_path)

    if not credits.empty:
        df = df.merge(credits, on=id_col, how="left")
    if "actors" not in df.columns:
        df["actors"] = [[]] * len(df)
    else:
        df["actors"] = df["actors"].apply(lambda x: x if isinstance(x, list) else [])

    if not keywords.empty:
        df = df.merge(keywords, on=id_col, how="left")
    if "keywords" not in df.columns:
        df["keywords"] = [[]] * len(df)
    else:
        df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])

    return df
