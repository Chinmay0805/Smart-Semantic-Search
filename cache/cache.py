
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional



@dataclass
class CacheEntry:
    """
    Represents a single cached query-result pair.

    Fields:
    - query:            original query string
    - embedding:        L2-normalized query embedding (numpy array)
    - result:           the answer/document returned for this query
    - dominant_cluster: which GMM cluster this query belongs to
    - timestamp:        when this entry was cached (unix time)
    """
    query:             str
    embedding:         np.ndarray
    result:            str
    dominant_cluster:  int
    timestamp:         float = field(default_factory=time.time)



class SemanticCache:
    """
    A cluster-partitioned semantic cache.

    Instead of storing all entries in one flat list,
    entries are grouped by their dominant GMM cluster.

    Lookup flow:
    1. Embed the incoming query
    2. Find its dominant cluster (via GMM)
    3. Only scan entries in THAT cluster
    4. Return the best match if similarity >= threshold

    This means lookup cost is O(n/k) where:
    - n = total cache entries
    - k = number of clusters (10 in our case)
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: minimum cosine similarity to count as a cache hit.
                                  Explored in cache/threshold_analysis.py
        """
        self.threshold = similarity_threshold

        # Cluster-partitioned storage
        # Key: cluster_id (int)
        # Value: list of CacheEntry objects
        self._store: dict[int, list[CacheEntry]] = defaultdict(list)

        # Stats counters
        self._hit_count  = 0
        self._miss_count = 0

        # Thread safety
        self._lock = threading.Lock()


    def lookup(
        self,
        query_embedding: np.ndarray,
        dominant_cluster: int
    ) -> tuple[Optional[CacheEntry], float]:
        """
        Search for a semantically similar cached query.

        Args:
            query_embedding:  L2-normalized embedding of incoming query
            dominant_cluster: GMM cluster index for this query

        Returns:
            (best_matching_entry, similarity_score)
            If no match found: (None, best_score_seen)
        """
        with self._lock:
            candidates = self._store[dominant_cluster]

            if not candidates:
                self._miss_count += 1
                return None, 0.0

            best_entry = None
            best_score = 0.0

            for entry in candidates:
                # Cosine similarity = dot product of L2-normalized vectors
                # We normalized at embedding time, so this is exact cosine sim
                score = float(np.dot(query_embedding, entry.embedding))

                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_score >= self.threshold:
                self._hit_count += 1
                return best_entry, best_score
            else:
                self._miss_count += 1
                return None, best_score


    def store(self, entry: CacheEntry) -> None:
        """
        Add a new entry to the cache under its dominant cluster.

        Args:
            entry: CacheEntry to store
        """
        with self._lock:
            self._store[entry.dominant_cluster].append(entry)

   
    def flush(self) -> None:
        """
        Wipe all cache entries and reset all stats counters.
        Called by DELETE /cache endpoint.
        """
        with self._lock:
            self._store.clear()
            self._hit_count  = 0
            self._miss_count = 0

  
    @property
    def stats(self) -> dict:
        """
        Returns current cache statistics.
        Called by GET /cache/stats endpoint.
        """
        with self._lock:
            total_entries = sum(len(v) for v in self._store.values())
            total_queries = self._hit_count + self._miss_count
            hit_rate      = (
                round(self._hit_count / total_queries, 3)
                if total_queries > 0 else 0.0
            )
            return {
                "total_entries": total_entries,
                "hit_count":     self._hit_count,
                "miss_count":    self._miss_count,
                "hit_rate":      hit_rate
            }

    @property
    def total_entries(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._store.values())

    def get_cluster_sizes(self) -> dict[int, int]:
        """Returns how many entries are in each cluster."""
        with self._lock:
            return {k: len(v) for k, v in self._store.items()}

    def __repr__(self):
        return (
            f"SemanticCache("
            f"threshold={self.threshold}, "
            f"entries={self.total_entries}, "
            f"hits={self._hit_count}, "
            f"misses={self._miss_count})"
        )