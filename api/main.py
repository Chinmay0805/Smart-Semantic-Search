"""
Part 4 - FastAPI Service
========================
Goal: Expose the semantic search system as a REST API.

Endpoints:
- POST   /query        — semantic search with cache
- GET    /cache/stats  — cache performance stats
- DELETE /cache        — flush cache and reset stats

Design decisions:
- Models loaded once at startup via lifespan (not on every request)
  → SentenceTransformer and GMM are expensive to load — do it once
- SemanticCache is a module-level singleton
  → Shared across all requests, maintains state between calls
- ChromaDB queried only on cache miss
  → Avoids vector search cost when cache can serve the answer
"""

import os
import sys
import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import chromadb

# Add project root to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cache.cache import SemanticCache, CacheEntry
from api.models import QueryRequest, QueryResponse, CacheStats, FlushResponse


# CONFIG

EMBED_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH       = "./embeddings/chroma_db"
GMM_MODEL_PATH    = "./models/gmm_model.joblib"
PCA_MODEL_PATH    = "./models/pca_model.joblib"
SIMILARITY_THRESHOLD = 0.60
TOP_K_RESULTS     = 1   # number of ChromaDB results to fetch on cache miss


# GLOBAL STATE
# (loaded once at startup, shared across requests)
embed_model  = None
gmm_model    = None
pca_model    = None
chroma_collection = None
cache        = SemanticCache(similarity_threshold=SIMILARITY_THRESHOLD)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    Loads all heavy models into memory so requests are fast.
    """
    global embed_model, gmm_model, pca_model, chroma_collection

    print("\n Starting Trademarkia Semantic Search API...")
    print("=" * 50)

    # Load embedding model
    print(" Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"    Loaded: {EMBED_MODEL_NAME}")

    # Load GMM clustering model
    print(" Loading GMM clustering model...")
    if not os.path.exists(GMM_MODEL_PATH):
        raise RuntimeError(
            f"GMM model not found at {GMM_MODEL_PATH}. "
            "Please run: python models/clustering.py"
        )
    gmm_model = joblib.load(GMM_MODEL_PATH)
    print(f"    Loaded GMM: {gmm_model.n_components} clusters")

    # Load PCA model
    print(" Loading PCA model...")
    if not os.path.exists(PCA_MODEL_PATH):
        raise RuntimeError(
            f"PCA model not found at {PCA_MODEL_PATH}. "
            "Please run: python models/clustering.py"
        )
    pca_model = joblib.load(PCA_MODEL_PATH)
    print(f"    Loaded PCA: {pca_model.n_components_} components")

    # Load ChromaDB collection
    print("  Loading ChromaDB collection...")
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_PATH}. "
            "Please run: python embeddings/build_index.py"
        )
    chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_collection("newsgroups")
    print(f"    Loaded ChromaDB: {chroma_collection.count()} documents")

    print("=" * 50)
    print(f" API ready! Semantic cache threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Docs: http://localhost:8000/docs")
    print("=" * 50)

    yield  # server runs here

    # Cleanup on shutdown
    print("\n Shutting down API...")



app = FastAPI(
    title="Trademarkia Semantic Search",
    description=(
        "A semantic search system over the 20 Newsgroups dataset. "
        "Features fuzzy GMM clustering and a smart semantic cache "
        "that recognizes similar queries even when phrased differently."
    ),
    version="1.0.0",
    lifespan=lifespan
)



def get_dominant_cluster(embedding: np.ndarray) -> int:
    """
    Given a query embedding, find which GMM cluster it belongs to most.

    Steps:
    1. Reduce embedding from 384D to 50D using PCA
    2. Get soft cluster probabilities from GMM
    3. Return the cluster with highest probability

    This is the same cluster partitioning used by the cache —
    so the query is compared only against cached entries in the same cluster.
    """
    reduced   = pca_model.transform([embedding])          # (1, 50)
    probs     = gmm_model.predict_proba(reduced)[0]        # (n_clusters,)
    return int(np.argmax(probs))



def search_chromadb(embedding: np.ndarray) -> str:
    """
    Query ChromaDB for the most semantically similar document.
    Called only on cache miss.
    """
    results = chroma_collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=TOP_K_RESULTS,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant documents found."

    # Return the top matching document
    return results["documents"][0][0]



# ENDPOINT 1: POST /query
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main search endpoint.

    Flow:
    1. Embed the incoming query
    2. Find its dominant GMM cluster
    3. Check semantic cache (only scan same cluster = fast)
    4a. Cache HIT  → return cached result immediately
    4b. Cache MISS → search ChromaDB, store in cache, return result
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1: Embed query
    # normalize_embeddings=True so dot product = cosine similarity
    embedding = embed_model.encode(
        request.query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Step 2: Find dominant cluster
    dominant_cluster = get_dominant_cluster(embedding)

    # Step 3: Check cache
    matched_entry, similarity_score = cache.lookup(embedding, dominant_cluster)

    # Step 4a: Cache HIT
    if matched_entry is not None:
        return QueryResponse(
            query            = request.query,
            cache_hit        = True,
            matched_query    = matched_entry.query,
            similarity_score = round(similarity_score, 4),
            result           = matched_entry.result,
            dominant_cluster = dominant_cluster
        )

    # Step 4b: Cache MISS — search ChromaDB
    result_text = search_chromadb(embedding)

    # Store in cache for future similar queries
    cache.store(CacheEntry(
        query            = request.query,
        embedding        = embedding,
        result           = result_text,
        dominant_cluster = dominant_cluster
    ))

    return QueryResponse(
        query            = request.query,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = round(similarity_score, 4),
        result           = result_text,
        dominant_cluster = dominant_cluster
    )


# ENDPOINT 2: GET /cache/stats
@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """
    Returns current cache performance statistics.
    Useful for monitoring how well the cache is working.
    """
    return CacheStats(**cache.stats)


# ENDPOINT 3: DELETE /cache
@app.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    """
    Wipes all cache entries and resets hit/miss counters.
    """
    cache.flush()
    return FlushResponse(
        status  = "cache flushed",
        message = "All entries cleared and stats reset"
    )


# ROOT — health check
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status":    "running",
        "service":   "Trademarkia Semantic Search",
        "version":   "1.0.0",
        "cache":     cache.stats,
        "docs":      "http://localhost:8000/docs"
    }