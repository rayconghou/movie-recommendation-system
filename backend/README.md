# Movie Recommendation Backend

This is a simple FastAPI backend that serves movie recommendations based on
TF-IDF vectors built from `movies_metadata.csv`.

## Setup

From the project root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Make sure `movies_metadata.csv` is present in the project root:

```text
cs_projects/movie_recommendation_system/
  movies_metadata.csv
  backend/
    main.py
    requirements.txt
```

## Running the server

From the `backend` directory with the virtual environment activated:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The main endpoint is:

- `POST /recommend` — body: `{\"query\": \"your natural language query\"}`.

Health check:

- `GET /health`

## Database and deterministic filters

The backend uses an **SQLite database** (created at `../data/movies.db` on first run) so that **deterministic filters** can narrow the search before TF-IDF ranking.

- On startup, if the DB is empty, it is populated from `movies_metadata.csv` (same cleaning and `release_year` logic). No extra setup is required.
- Queries are parsed for **release decade/year** with **deterministic rules** (no LLM):
  - e.g. **"90s"**, **"1990s"** → only movies from 1990–1999
  - **"80s"**, **"2000s"** → 1980–1989, 2000–2009
  - **"early 2000s"** → 2000–2004, **"late 90s"** → 1995–1999
- Example: *"movie from the 90s where ship hits iceberg"* first restricts to 1990s, then ranks by TF-IDF similarity to *"movie where ship hits iceberg"* among those candidates only.

The DB schema is in `database.py`; the table is indexed on `release_year` for fast filtering.
