"use client";

import { FormEvent, useState } from "react";
import { Search } from "lucide-react";
import PipelineProgress from "@/components/PipelineProgress";
import type { QueryResponse } from "@/services/api";

const EXAMPLE_QUERIES = [
  "NASA shuttle missions",
  "Graphics driver updates",
  "encryption clipper chip NSA",
] as const;

interface SearchPanelProps {
  query: string;
  onQueryChange: (value: string) => void;
  onSearch: (query: string) => void;
  isLoading: boolean;
  response: QueryResponse | null;
  error: string | null;
  backendOnline: boolean | null;
}

function ConnectionStatus({ online }: { online: boolean | null }) {
  if (online === null) {
    return (
      <span className="text-sm text-slate-500">
        <span className="text-slate-600">●</span> Checking…
      </span>
    );
  }

  if (online) {
    return (
      <span className="text-sm text-slate-400">
        <span className="text-green-500">●</span> Connected
      </span>
    );
  }

  return (
    <span className="text-sm text-slate-400">
      <span className="text-red-500">●</span> Offline
    </span>
  );
}

export default function SearchPanel({
  query,
  onQueryChange,
  onSearch,
  isLoading,
  response,
  error,
  backendOnline,
}: SearchPanelProps) {
  const [expanded, setExpanded] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSearch(query);
  };

  const handleExampleClick = (example: string) => {
    onQueryChange(example);
    onSearch(example);
  };

  const resultPreview =
    response?.result && response.result.length > 2000 && !expanded
      ? `${response.result.slice(0, 2000)}…`
      : response?.result;

  return (
    <div className="space-y-8">
      <section>
        <div className="mb-4 flex items-center justify-between">
          <ConnectionStatus online={backendOnline} />
        </div>

        <form onSubmit={handleSubmit}>
          <div className="relative">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              placeholder="Search 20 Newsgroups…"
              disabled={isLoading}
              className="h-14 w-full rounded-2xl border border-slate-800 bg-slate-900 pl-12 pr-28 text-base text-slate-100 placeholder:text-slate-600 outline-none transition focus:border-slate-700 disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? "Searching…" : "Search"}
            </button>
          </div>
        </form>

        <p className="mt-3 text-sm text-slate-500">
          Try these:{" "}
          {EXAMPLE_QUERIES.map((example, index) => (
            <span key={example}>
              {index > 0 && (
                <span className="mx-1.5 text-slate-700" aria-hidden>
                  •
                </span>
              )}
              <button
                type="button"
                onClick={() => handleExampleClick(example)}
                disabled={isLoading}
                className="text-slate-400 transition hover:text-blue-500 disabled:opacity-50"
              >
                {example}
              </button>
            </span>
          ))}
        </p>
      </section>

      {error && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900 px-4 py-3 text-sm text-red-500">
          {error}
        </div>
      )}

      <PipelineProgress isActive={isLoading} />

      {!isLoading && response && (
        <section className="rounded-2xl border border-slate-800 bg-slate-900">
          <div className="border-b border-slate-800 px-5 py-3">
            <h2 className="text-sm font-medium text-slate-300">Result</h2>
            <p className="mt-0.5 text-xs text-slate-500">
              Top matching document from ChromaDB
            </p>
          </div>
          <pre className="max-h-[32rem] overflow-auto p-5 font-mono text-sm leading-relaxed whitespace-pre-wrap text-slate-300">
            {resultPreview}
          </pre>
          {response.result.length > 2000 && (
            <div className="border-t border-slate-800 px-5 py-3">
              <button
                type="button"
                onClick={() => setExpanded((value) => !value)}
                className="text-sm text-blue-500 hover:text-blue-400"
              >
                {expanded ? "Show less" : "Show full document"}
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
