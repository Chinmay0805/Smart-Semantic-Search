"""
Pydantic Models for FastAPI
===========================
Defines request and response schemas for all API endpoints.
Pydantic validates incoming JSON automatically — no manual checks needed.
"""

from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    """
    Request body for POST /query
    Example: {"query": "What are NASA's latest space missions?"}
    """
    query: str


class QueryResponse(BaseModel):
    """
    Response body for POST /query

    Fields:
    - query:            the original query string sent by user
    - cache_hit:        True if a similar query was found in cache
    - matched_query:    the cached query that matched (None on miss)
    - similarity_score: cosine similarity between queries (0.0 on miss)
    - result:           the answer/document text
    - dominant_cluster: which GMM cluster this query belongs to
    """
    query:             str
    cache_hit:         bool
    matched_query:     Optional[str]
    similarity_score:  float
    result:            str
    dominant_cluster:  int


class CacheStats(BaseModel):
    """
    Response body for GET /cache/stats
    """
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float


class FlushResponse(BaseModel):
    """
    Response body for DELETE /cache
    """
    status:  str
    message: str