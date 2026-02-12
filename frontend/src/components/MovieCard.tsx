// import Image from "next/image";

export type Movie = {
  rank: number;
  title: string;
  poster_path: string | null;
  overview_truncated: string | null;
  genres: string[];
  release_year: number | null;
  score: number;
  justification: string;
};

// const TMDB_BASE_URL = "https://image.tmdb.org/t/p/w200";

type MovieCardProps = {
  movie: Movie;
};

export function MovieCard({ movie }: MovieCardProps) {
  // const posterUrl =
  //   movie.poster_path != null && movie.poster_path.trim().length > 0
  //     ? `${TMDB_BASE_URL}${movie.poster_path}`
  //     : null;

  return (
    <article className="flex gap-4 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex w-32 flex-shrink-0 flex-col items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
          Rank
        </span>
        <span className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
          #{movie.rank}
        </span>
        <h2 className="mt-1 line-clamp-3 text-center text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          {movie.title}
          {movie.release_year != null && (
            <span className="ml-1 text-xs font-normal text-zinc-500 dark:text-zinc-400">
              ({movie.release_year})
            </span>
          )}
        </h2>

        {/* Poster rendering is intentionally disabled for now.
        {posterUrl && (
          <div className="relative h-40 w-full overflow-hidden rounded-lg bg-zinc-100 dark:bg-zinc-800">
            <Image
              src={posterUrl}
              alt={movie.title}
              fill
              sizes="160px"
              className="object-cover"
            />
          </div>
        )} */}
      </div>

      <div className="flex flex-1 flex-col gap-2">
        {movie.overview_truncated && (
          <p className="text-sm text-zinc-700 dark:text-zinc-300">
            {movie.overview_truncated}
          </p>
        )}

        {movie.genres.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1.5">
            {movie.genres.map((genre) => (
              <span
                key={genre}
                className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700 dark:bg-indigo-950 dark:text-indigo-200"
              >
                {genre}
              </span>
            ))}
          </div>
        )}

        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          {movie.justification}
        </p>

        <span className="mt-auto text-xs text-zinc-400 dark:text-zinc-500">
          Similarity score: {movie.score.toFixed(3)}
        </span>
      </div>
    </article>
  );
}

