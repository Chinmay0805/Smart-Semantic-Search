"use client";

import { useCallback, useEffect, useState } from "react";
import SearchPanel from "@/components/SearchPanel";
import DiagnosticsDrawer from "@/components/DiagnosticsDrawer";
import {
  checkHealth,
  getCacheStats,
  searchQuery,
  flushCache,
  type CacheStats,
  type HealthStatus,
  type QueryResponse,
} from "@/services/api";

export default function Home() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const refreshCacheStats = useCallback(async () => {
    const stats = await getCacheStats();
    setCacheStats(stats);
    return stats;
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const [healthData, stats] = await Promise.all([
          checkHealth(),
          getCacheStats(),
        ]);

        if (cancelled) return;

        setHealth(healthData);
        setCacheStats(stats);
        setBackendOnline(true);
      } catch {
        if (!cancelled) {
          setBackendOnline(false);
        }
      }
    }

    bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleSearch = async (searchText: string) => {
    setIsLoading(true);
    setError(null);
    setLatencyMs(null);

    const startedAt = performance.now();

    try {
      const result = await searchQuery(searchText);
      setResponse(result);
      setLatencyMs(Math.round(performance.now() - startedAt));
      await refreshCacheStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed.");
      setResponse(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFlushCache = async () => {
    await flushCache();
    const stats = await refreshCacheStats();
    setCacheStats(stats);
    setResponse(null);
    setLatencyMs(null);
  };

  return (
    <div className="min-h-full bg-[#0B1220] text-slate-100">
      <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6 sm:py-16">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
            Smart Semantic Search
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-base text-slate-400">
            Semantic retrieval over 20 Newsgroups with GMM-partitioned caching
            and ChromaDB vector search.
          </p>
          {health && (
            <p className="mt-2 text-xs text-slate-600">
              {health.service} v{health.version}
            </p>
          )}
        </header>

        <main className="space-y-6">
          <SearchPanel
            query={query}
            onQueryChange={setQuery}
            onSearch={handleSearch}
            isLoading={isLoading}
            response={response}
            error={error}
            backendOnline={backendOnline}
          />

          {response && !isLoading && (
            <DiagnosticsDrawer
              response={response}
              cacheStats={cacheStats}
              latencyMs={latencyMs}
              onFlushCache={handleFlushCache}
            />
          )}
        </main>

        <footer className="mt-16 border-t border-slate-800 pt-6 text-center text-xs text-slate-600">
          Embed → PCA → GMM routing → Semantic cache → ChromaDB
        </footer>
      </div>
    </div>
  );
}
