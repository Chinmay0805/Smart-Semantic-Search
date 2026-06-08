"use client";

import { useState, type ReactNode } from "react";
import { ChevronDown, Trash2 } from "lucide-react";
import type { CacheStats, QueryResponse } from "@/services/api";

const CLUSTER_LABELS: Record<number, string> = {
  0: "Marketplace",
  1: "Space & Science",
  2: "Religion",
  3: "Sports",
  4: "Politics & Law",
  5: "Hardware",
  6: "Cryptography",
  7: "Software",
  8: "Middle East",
  9: "Vehicles",
};

interface DiagnosticsDrawerProps {
  response: QueryResponse;
  cacheStats: CacheStats | null;
  latencyMs: number | null;
  onFlushCache: () => Promise<void>;
}

function DiagnosticRow({
  label,
  value,
}: {
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-800 py-3 last:border-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-right text-sm text-slate-200">{value}</span>
    </div>
  );
}

export default function DiagnosticsDrawer({
  response,
  cacheStats,
  latencyMs,
  onFlushCache,
}: DiagnosticsDrawerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isFlushing, setIsFlushing] = useState(false);
  const [flushMessage, setFlushMessage] = useState<string | null>(null);
  const [flushError, setFlushError] = useState<string | null>(null);

  const clusterLabel =
    CLUSTER_LABELS[response.dominant_cluster] ?? "Unknown";

  const handleFlush = async () => {
    setIsFlushing(true);
    setFlushMessage(null);
    setFlushError(null);

    try {
      await onFlushCache();
      setFlushMessage("Cache flushed successfully.");
    } catch (err) {
      setFlushError(
        err instanceof Error ? err.message : "Failed to flush cache."
      );
    } finally {
      setIsFlushing(false);
    }
  };

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900">
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        className="flex w-full items-center justify-between px-5 py-4 text-left"
        aria-expanded={isOpen}
      >
        <div>
          <h2 className="text-sm font-medium text-slate-200">
            Search Diagnostics
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Cache routing, similarity, and cluster metadata
          </p>
        </div>
        <ChevronDown
          className={`h-4 w-4 text-slate-500 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && (
        <div className="border-t border-slate-800 px-5 pb-5">
          <div className="pt-1">
            <DiagnosticRow
              label="Cache status"
              value={
                <span
                  className={
                    response.cache_hit ? "text-blue-500" : "text-slate-400"
                  }
                >
                  {response.cache_hit ? "Hit" : "Miss"}
                </span>
              }
            />
            {response.cache_hit && response.matched_query && (
              <DiagnosticRow
                label="Matched query"
                value={
                  <span className="max-w-xs font-mono text-xs text-slate-300">
                    {response.matched_query}
                  </span>
                }
              />
            )}
            <DiagnosticRow
              label="Similarity score"
              value={response.similarity_score.toFixed(4)}
            />
            <DiagnosticRow
              label="GMM cluster"
              value={`${response.dominant_cluster} — ${clusterLabel}`}
            />
            {latencyMs !== null && (
              <DiagnosticRow label="Latency" value={`${latencyMs} ms`} />
            )}
            {cacheStats && (
              <>
                <DiagnosticRow
                  label="Cache entries"
                  value={cacheStats.total_entries}
                />
                <DiagnosticRow
                  label="Global hit rate"
                  value={`${(cacheStats.hit_rate * 100).toFixed(1)}%`}
                />
                <DiagnosticRow
                  label="Hits / misses"
                  value={`${cacheStats.hit_count} / ${cacheStats.miss_count}`}
                />
              </>
            )}
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-slate-800 pt-4">
            <button
              type="button"
              onClick={handleFlush}
              disabled={isFlushing}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-red-500 transition hover:border-red-500/50 hover:bg-slate-800 disabled:opacity-50"
            >
              <Trash2 className="h-3.5 w-3.5" />
              {isFlushing ? "Flushing…" : "Flush cache"}
            </button>
            {flushMessage && (
              <span className="text-xs text-slate-400">{flushMessage}</span>
            )}
            {flushError && (
              <span className="text-xs text-red-500">{flushError}</span>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
