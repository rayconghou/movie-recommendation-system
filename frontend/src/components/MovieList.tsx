import { Movie, MovieCard } from "./MovieCard";

type MovieListProps = {
  movies: Movie[];
};

export function MovieList({ movies }: MovieListProps) {
  if (movies.length === 0) {
    return (
      <div className="mt-10 rounded-xl border border-dashed border-zinc-300 bg-zinc-50 p-8 text-center text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-400">
        No results yet. Try describing a movie you&apos;re in the mood for.
      </div>
    );
  }

  return (
    <div className="mt-6 h-[70vh] w-full overflow-y-auto rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex flex-col gap-4">
        {movies.map((movie) => (
          <MovieCard key={movie.rank} movie={movie} />
        ))}
      </div>
    </div>
  );
}

