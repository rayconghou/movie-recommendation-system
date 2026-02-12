# Movie Recommendation System

A full-stack movie recommendation app: natural-language search with TF-IDF ranking, optional decade filters (SQLite), and LLM-generated explanations.

- **Frontend**: Next.js 13 (App Router) + Tailwind — search bar and scrollable list of top 30 movies with rank, title, metadata, and justification.
- **Backend**: FastAPI + scikit-learn — loads `movies_metadata.csv`, builds TF-IDF vectors, uses an SQLite DB for deterministic filters (e.g. "90s" → 1990–1999), and optionally calls OpenAI to explain why each movie is ranked.

## Prerequisites

- **movies_metadata.csv** in the project root. Not committed (see `.gitignore`); download e.g. from [TMDB metadata on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place it in the repo root.
- **Node.js** and **npm** for the frontend.
- **Python 3.10+** for the backend.
- **OpenAI API key** (optional): set `OPENAI_API_KEY` for AI-generated ranking explanations.

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will create `../data/movies.db` from the CSV on first run. Optional: create a `.env` in the project root with `OPENAI_API_KEY=sk-...` and `OPENAI_MODEL=gpt-4o-mini` (or another model).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Set `BACKEND_URL=http://127.0.0.1:8000` in `frontend/.env.local` if the backend runs elsewhere.

### 3. Use it

Type a query like *"movie from the 90s where ship hits iceberg"*. Results are restricted to the 1990s (deterministic filter), then ranked by TF-IDF similarity; each result shows an LLM justification when `OPENAI_API_KEY` is set.

## Project layout

```
movie_recommendation_system/
  movies_metadata.csv     # Required: TMDB-style metadata
  data/
    movies.db            # Created by backend (SQLite, from CSV)
  backend/               # FastAPI + TF-IDF + DB filters + OpenAI justification
    main.py
    database.py
    requirements.txt
  frontend/              # Next.js app
    src/app/
    src/components/
```

## License

MIT
