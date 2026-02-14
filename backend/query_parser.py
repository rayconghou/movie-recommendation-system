"""
LLM-based query parser: extract structured slots (genres, keywords, actors, decade, free_text, fields_to_use)
from a natural-language movie search query.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Allowed field names for vectorization
ALLOWED_FIELDS = {"genres", "keywords", "overview", "actors", "title"}


class QueryIntent(BaseModel):
    """Structured extraction from a movie search query."""

    genres: List[str] = []
    keywords: List[str] = []
    featured_actors: List[str] = []
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    free_text: str = ""
    fields_to_use: List[str] = []


def parse_query_for_search(query: str) -> QueryIntent:
    """
    Use the LLM to extract structured slots from the user's movie search query.
    Returns QueryIntent; on API failure or missing key, returns a default intent
    with free_text=query and fields_to_use=["overview", "title", "genres"].
    """
    if not query or not query.strip():
        return QueryIntent(free_text="", fields_to_use=["overview", "title", "genres"])

    if not OPENAI_API_KEY.strip():
        return QueryIntent(
            free_text=query.strip(),
            fields_to_use=["overview", "title", "genres"],
        )

    system = (
        "You extract structured movie search intent from the user's query. "
        "Return ONLY a JSON object with these keys (use empty lists or null when not specified):\n"
        "- genres: list of genre names the user wants (e.g. Comedy, Drama, Action)\n"
        "- keywords: list of thematic/keyword terms (e.g. ship, iceberg, heist)\n"
        "- featured_actors: list of actor names mentioned\n"
        "- start_year, end_year: integers or null for release year range (inclusive)\n"
        "- free_text: string of the remaining unstructured part for semantic search\n"
        "- fields_to_use: list of which fields to use for matching, from: genres, keywords, overview, actors, title. "
        "Include the fields that are relevant given what the user asked for (e.g. if they mention actors include 'actors').\n"
        "No explanation, only the JSON object."
    )

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=400,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            return _default_intent(query)
        # Strip markdown code block if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
        genres = data.get("genres") or []
        keywords = data.get("keywords") or []
        featured_actors = data.get("featured_actors") or []
        start_year = data.get("start_year")
        end_year = data.get("end_year")
        free_text = data.get("free_text") or query.strip()
        fields_to_use = data.get("fields_to_use") or []
        if not isinstance(genres, list):
            genres = []
        if not isinstance(keywords, list):
            keywords = []
        if not isinstance(featured_actors, list):
            featured_actors = []
        if not isinstance(fields_to_use, list):
            fields_to_use = []
        fields_to_use = [f for f in fields_to_use if f in ALLOWED_FIELDS]
        if not fields_to_use:
            fields_to_use = ["overview", "title", "genres"]
        if start_year is not None and not isinstance(start_year, int):
            start_year = None
        if end_year is not None and not isinstance(end_year, int):
            end_year = None
        return QueryIntent(
            genres=genres,
            keywords=keywords,
            featured_actors=featured_actors,
            start_year=start_year,
            end_year=end_year,
            free_text=free_text,
            fields_to_use=fields_to_use,
        )
    except Exception:
        return _default_intent(query)


def _default_intent(query: str) -> QueryIntent:
    return QueryIntent(
        free_text=query.strip(),
        fields_to_use=["overview", "title", "genres"],
    )
