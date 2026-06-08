const DEFAULT_BASE_URL = "http://localhost:8000";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? DEFAULT_BASE_URL;

export interface QueryParams {
  query: string;
}

export interface QueryResponse {
  query: string;
  cache_hit: boolean;
  matched_query: string | null;
  similarity_score: number;
  result: string;
  dominant_cluster: number;
}

export interface CacheStats {
  total_entries: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  cache: CacheStats;
  docs: string;
}

export interface FlushCacheResponse {
  status: string;
  message: string;
}

interface FastAPIValidationError {
  type: string;
  loc: (string | number)[];
  msg: string;
}

interface FastAPIErrorBody {
  detail?: string | FastAPIValidationError[];
}

function parseErrorDetail(body: FastAPIErrorBody): string | undefined {
  const { detail } = body;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((item) => item.msg).join("; ");
  }

  return undefined;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const body = (await response.json()) as FastAPIErrorBody;
      const detail = parseErrorDetail(body);

      if (detail) {
        message = detail;
      }
    } catch {
      // Response body was not JSON; keep the default message.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export async function checkHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/");
}

export async function searchQuery(query: string): Promise<QueryResponse> {
  const trimmedQuery = query.trim();

  if (!trimmedQuery) {
    throw new Error("Query cannot be empty");
  }

  return request<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify({ query: trimmedQuery } satisfies QueryParams),
  });
}

export async function getCacheStats(): Promise<CacheStats> {
  return request<CacheStats>("/cache/stats");
}

export async function flushCache(): Promise<FlushCacheResponse> {
  return request<FlushCacheResponse>("/cache", {
    method: "DELETE",
  });
}

export const api = {
  checkHealth,
  searchQuery,
  getCacheStats,
  flushCache,
};

export { API_BASE_URL };
