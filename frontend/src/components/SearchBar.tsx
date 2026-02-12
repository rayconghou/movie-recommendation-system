"use client";

import { FormEvent } from "react";

type SearchBarProps = {
  query: string;
  onQueryChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
};

export function SearchBar({
  query,
  onQueryChange,
  onSubmit,
  isLoading,
}: SearchBarProps) {
  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit();
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full max-w-3xl items-center gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-2 shadow-sm focus-within:ring-2 focus-within:ring-indigo-500 dark:border-zinc-800 dark:bg-zinc-900"
    >
      <input
        type="text"
        placeholder="Describe the kind of movie you want to watch…"
        className="flex-1 bg-transparent text-sm text-zinc-900 placeholder:text-zinc-400 outline-none dark:text-zinc-100"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
      />
      <button
        type="submit"
        disabled={isLoading}
        className="inline-flex items-center rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isLoading ? "Searching…" : "Search"}
      </button>
    </form>
  );
}

