 "use client";

import { useState } from "react";
import { MovieList } from "../components/MovieList";
import { SearchBar } from "../components/SearchBar";
import type { Movie } from "../components/MovieCard";

export default function Home() {
  const [query, setQuery] = useState("");
  const [movies, setMovies] = useState<Movie[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch() {
    if (!query.trim()) {
      setError("Please enter a description of what you want to watch.");
      setMovies([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const message =
          typeof data.error === "string"
            ? data.error
            : "Something went wrong while fetching recommendations.";
        throw new Error(message);
      }

      const data = await res.json();
      setMovies(data as Movie[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error.");
      setMovies([]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen justify-center bg-zinc-50 px-4 py-10 font-sans dark:bg-black">
      <main className="flex w-full max-w-5xl flex-col gap-6">
        <header className="flex flex-col gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Natural Language Movie Search
          </h1>
          <p className="max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
            Type a description like{" "}
            <span className="font-medium">
              “a slow-burn crime drama set in a big city”
            </span>{" "}
            and we&apos;ll rank movies from the dataset by semantic similarity.
          </p>
        </header>

        <SearchBar
          query={query}
          onQueryChange={setQuery}
          onSubmit={handleSearch}
          isLoading={isLoading}
        />

        {error && (
          <div className="mt-2 text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        )}

        <MovieList movies={movies} />
      </main>
    </div>
  );
}

