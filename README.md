---
title: Smart Semantic Search
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Smart Semantic Search System

A semantic search pipeline over the 20 Newsgroups dataset (~17,000 articles), featuring fuzzy clustering, a hand-built semantic cache, and a FastAPI service — fully containerised with Docker.

---

## Architecture
```
User Query
    ↓
[FastAPI]         ← receives the HTTP request
    ↓
[Semantic Cache]  ← recognises similar past queries
    ↓ (on cache miss)
[ChromaDB]        ← vector similarity search
    ↑
[GMM Clusters]    ← partitions cache for O(n/k) lookup
```

---

## Project Structure
```
Smart-Semantic-Search/
├── api/
│   ├── __init__.py            # package marker
│   ├── cache.py               # SemanticCache — built from scratch
│   ├── models.py              # Pydantic schemas
│   └── main.py                # FastAPI app
├── embeddings/
│   ├── build_index.py         # clean → embed → store pipeline
│   ├── chroma_db/             # persisted vector database
│   ├── embeddings.npy         # raw document embeddings
│   └── texts.json             # cleaned texts + metadata
├── models/
│   ├── clustering.py          # GMM fuzzy clustering
│   ├── gmm_model.joblib       # trained GMM
│   ├── pca_model.joblib       # PCA transformer
│   ├── gmm_assignments.npy    # soft cluster probabilities
│   └── bic_curve.png          # BIC/AIC model selection plot
├── cache/
│   ├── threshold_analysis.py  # threshold exploration
│   └── threshold_analysis.png # hit rate vs precision plot
├── dataset/                      # raw dataset (not committed)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore
```

---

## Quickstart (Local)

**1. Clone & setup**
```bash
git clone https://github.com/Chinmay0805/Smart-Semantic-Search.git
cd Smart-Semantic-Search
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

**2. Build the index**
```bash
python embeddings/build_index.py
```

**3. Run clustering**
```bash
python models/clustering.py
```

**4. Run threshold analysis**
```bash
python cache/threshold_analysis.py
```

**5. Start the API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API live at http://localhost:8000 — interactive docs at http://localhost:8000/docs

---

## Quickstart (Docker)
```bash
docker build -t trademarkia-semantic-search .
docker run -p 8000:8000 trademarkia-semantic-search
```

---

## API Reference

### `POST /query`
Embeds the query, checks the semantic cache, returns a cached result on hit or searches ChromaDB on miss.
```json
// Request
{ "query": "What are NASA's latest space missions?" }

// Cache miss response
{
  "query": "What are NASA's latest space missions?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": "...",
  "dominant_cluster": 1
}

// Cache hit response (different wording, same meaning)
{
  "query": "Recent space exploration news from NASA",
  "cache_hit": true,
  "matched_query": "What are NASA's latest space missions?",
  "similarity_score": 0.707,
  "result": "...",
  "dominant_cluster": 1
}
```

### `GET /cache/stats`
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`
Flushes all entries and resets counters.
```json
{ "status": "cache flushed", "message": "All entries cleared and stats reset" }
```

### `GET /`
Health check — returns service status and current cache stats.

---

## Design Decisions

### Part 1 — Embedding & Vector Database

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim dense vectors, best speed/quality tradeoff for semantic similarity, CPU-friendly |
| Vector DB | ChromaDB | Serverless, persists to disk, native cosine similarity |
| Min doc length | 100 chars | Docs shorter than this don't embed meaningfully |
| Remove headers/footers | Yes | Adds noise, not semantic signal |
| Result | 16,806 / 18,846 docs kept | 89% retention after cleaning |

---

### Part 2 — Fuzzy Clustering

**Why GMM instead of KMeans:**
In KMeans every document belongs to exactly one cluster.
GMM produces a probability distribution over clusters. A post about gun legislation
belongs ~60% to politics and ~40% to firearms. GMM captures this but KMeans cannot.

**Why k=10:**
Fit GMM for k ∈ [5, 40]. BIC was lowest at k=10 — marginal improvement dropped
below 1% of total BIC range beyond this point. Mathematical justification, not
a convenience choice. See `models/bic_curve.png`.

**PCA to 50 dimensions:**
Full covariance GMM on 384-dim vectors is computationally intractable.
PCA to 50 dims retains sufficient structure (49.5% variance) for clean separation
while making GMM feasible.

**Cluster results:**

| Cluster | Top Terms | Topic | Coherence |
|---------|-----------|-------|-----------|
| 0 | mail, sale, price, offer | Marketplace | 0.114 |
| 1 | space, orbit, nasa | Space & Science | 0.076 |
| 2 | god, jesus, bible, christian | Religion | 0.171 |
| 3 | game, team, hockey, baseball | Sports | 0.229 |
| 4 | people, gun, government, fbi | Politics & Law | 0.142 |
| 5 | drive, card, scsi, mac, disk | Hardware | 0.172 |
| 6 | key, encryption, clipper, nsa | Cryptography | 0.233 |
| 7 | windows, file, program, dos | Software | 0.131 |
| 8 | israel, jews, armenian, war | Middle East | 0.270 |
| 9 | car, bike, engine, miles | Vehicles | 0.124 |

Cluster 8 (Middle East) is tightest at 0.270 — very focused domain.
Cluster 1 (Space) is broadest at 0.076 — covers many sub-topics.

**Boundary documents:**
Doc 4494 scored 41% Religion / 40% Politics. This is ambiguous content
that a hard-label system would misclassify. These boundary cases are the most
semantically interesting and validate the fuzzy approach.

---

### Part 3 — Semantic Cache

**Cluster-partitioned storage:**
Entries are stored in `defaultdict(list)` keyed by dominant GMM cluster.
Lookup only scans the same cluster as the incoming query — O(n/k) average
cost instead of O(n). At 10,000 cached entries across 10 clusters, this
means scanning ~1,000 entries instead of 10,000.

**Similarity:** Cosine similarity via `numpy` dot product on L2-normalised vectors.
Normalisation happens at embedding time so lookup is a single dot product.

**Thread safety:** `threading.Lock` guards all reads and writes.
FastAPI is async and handles concurrent requests — without locking,
simultaneous cache writes would corrupt state.

**Threshold analysis:**

| Threshold | Hit Rate | Precision | Verdict |
|-----------|----------|-----------|---------|
| 0.50 | 0.62 | 1.00 |  Best balance |
| 0.65 | 0.25 | 1.00 |  Misses valid paraphrases |
| 0.80+ | 0.00 | 0.00 |  Cache never triggers |

For `all-MiniLM-L6-v2`, paraphrases score 0.60–0.77 and unrelated queries
score 0.04–0.10. Default threshold set to **0.60** based on this empirical analysis.
See `cache/threshold_analysis.png`.

---

### Part 4 — Docker

**Base image:** `python:3.11-slim` — full Python 3.11 with ~200MB smaller footprint than the standard image.

**CPU-only PyTorch:** Installed first via PyTorch's CPU wheel index. Avoids downloading GPU torch (915MB) + NVIDIA CUDA libraries (1.5GB total). CPU inference is fully sufficient here.

**onnxruntime on Linux:** Docker containers run Linux via WSL2, even on Windows hosts. The Windows `onnxruntime==1.16.3` package doesn't exist for Linux — `onnxruntime==1.18.1` is installed separately as the correct Linux-compatible version.

---

## Verified Results
```
Query 1: "What are NASA latest space missions?"    → cache_hit: false
Query 2: "Recent space exploration news from NASA" → cache_hit: true, similarity: 0.707

Cache stats:
  total_entries: 1 | hit_count: 2 | miss_count: 1 | hit_rate: 0.667
```

---

## Requirements

- Python 3.10+
- 4GB RAM minimum
- No GPU required
- Docker Desktop (for containerised deployment)

---

Deployement STEP :


Step 1: Prepare the Hugging Face Space
Log in to your Hugging Face account and create a New Space.

Name it (e.g., Semantic-Search).

Select Docker as the Space SDK and choose the Blank template.

Click Create Space.

Step 2: Configure Your Project Files
Before touching Git, ensure your core configuration files are ready for Hugging Face's environment.

1. Update Dockerfile
Hugging Face requires your app to run on port 7860. Ensure your final command looks like this:

Dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
2. Update README.md
Add the required YAML configuration block to the very top of your README.md file (starting on line 1):

YAML
---
title: Smart Semantic Search
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
3. Check Ignore Files
Ensure that embeddings/chroma_db/ is not listed in your .gitignore or .dockerignore files, so the pre-built database can be uploaded.

Step 3: Initialize Git and Git LFS
Open your terminal in the root directory of your project (where the Dockerfile is) and run these commands to set up Git Large File Storage for all your models, datasets, and hidden database files.

Bash
# Initialize Git and LFS
git init
git lfs install

# Track standard ML files
git lfs track "*.npy"
git lfs track "*.joblib"
git lfs track "*.sqlite3"
git lfs track "*.tar.gz"
git lfs track "*.json"

# Track hidden ChromaDB files
git lfs track "*.bin"
git lfs track "*.pickle"
Step 4: Commit and Push
Now, commit your code and push it to your Hugging Face Space using an Access Token as your password.

Bash
# 1. Commit the LFS tracking rules first
git add .gitattributes
git commit -m "Setup Git LFS rules"

# 2. Add and commit the rest of your application code
git add .
git commit -m "Add application code and models"

# 3. Rename branch to 'main' (if it defaulted to master)
git branch -M main

# 4. Link your local repository to Hugging Face
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# 5. Push to Hugging Face
git push -u huggingface main