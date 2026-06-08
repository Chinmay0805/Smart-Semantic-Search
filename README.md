---
title: Smart Semantic Search
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Smart Semantic Search

A full-stack semantic search system over the **20 Newsgroups** dataset (~16,800 cleaned articles). The backend combines **SentenceTransformer** embeddings, **GMM-partitioned semantic caching**, and **ChromaDB** vector retrieval behind a **FastAPI** service. A **Next.js** frontend provides a minimalist search UI with pipeline visualization and collapsible diagnostics.

> **Note:** This is retrieval-only — there is no LLM. On cache miss, the API returns the raw text of the most similar Usenet document from ChromaDB.

---

## Architecture

```
Browser (Next.js :3000)
    ↓  REST / JSON
[FastAPI :8000]
    ↓
Embed query (all-MiniLM-L6-v2)
    ↓
PCA → GMM dominant cluster
    ↓
[Semantic Cache]  ← cluster-partitioned, cosine similarity ≥ 0.60
    ↓ (on miss)
[ChromaDB]        ← top-1 cosine vector search
```

---

## Project Structure

```
Smart-Semantic-Search/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app, CORS, endpoints
│   │   └── models.py            # Pydantic request/response schemas
│   ├── cache/
│   │   ├── cache.py             # SemanticCache (built from scratch)
│   │   └── threshold_analysis.py
│   ├── embeddings/
│   │   ├── build_index.py       # clean → embed → ChromaDB pipeline
│   │   ├── chroma_db/           # persisted vector database
│   │   ├── embeddings.npy
│   │   └── texts.json
│   ├── models/
│   │   ├── clustering.py        # PCA + GMM training
│   │   ├── gmm_model.joblib
│   │   ├── pca_model.joblib
│   │   └── bic_curve.png
│   ├── dataset/                 # raw 20 Newsgroups data
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # main search page
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── SearchPanel.tsx
│   │   │   ├── PipelineProgress.tsx
│   │   │   └── DiagnosticsDrawer.tsx
│   │   └── services/
│   │       └── api.ts           # typed FastAPI client
│   └── package.json
└── README.md
```

---

## Quickstart (Full Stack — Local)

### Prerequisites

- Python 3.10+
- Node.js 18+
- 4 GB RAM minimum
- No GPU required

### 1. Clone the repository

```bash
git clone https://github.com/Chinmay0805/Smart-Semantic-Search.git
cd Smart-Semantic-Search
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

If the pre-built index and models are not present, run the offline pipelines from the `backend/` directory:

```bash
python embeddings/build_index.py
python models/clustering.py
python cache/threshold_analysis.py   # optional — generates analysis plots
```

Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### 3. Frontend setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

- UI: http://localhost:3000

The frontend defaults to `http://localhost:8000` for API calls. Override with a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

CORS is configured on the backend to allow `http://localhost:3000`.

---

## Frontend

Built with **Next.js 15** (App Router), **TypeScript**, and **Tailwind CSS v4**.

| Feature | Description |
|---------|-------------|
| Search hero | Large 56px input with example query links |
| Pipeline progress | Step indicator during search: Embed → GMM → Cache → ChromaDB |
| Results | Raw Usenet document text in a monospace panel |
| Diagnostics drawer | Collapsible panel: cache hit/miss, similarity, cluster, latency, flush cache |
| Connection status | `● Connected` / `● Offline` indicator |

### Frontend scripts

```bash
npm run dev      # development server
npm run build    # production build
npm run start    # serve production build
npm run lint     # ESLint
```

---

## API Reference

Base URL (local): `http://localhost:8000`

### `GET /` — Health check

```json
{
  "status": "running",
  "service": "Trademarkia Semantic Search",
  "version": "1.0.0",
  "cache": {
    "total_entries": 0,
    "hit_count": 0,
    "miss_count": 0,
    "hit_rate": 0.0
  },
  "docs": "http://localhost:8000/docs"
}
```

### `POST /query` — Semantic search

**Request:**

```json
{ "query": "What are NASA's latest space missions?" }
```

**Cache miss response:**

```json
{
  "query": "What are NASA's latest space missions?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": "...",
  "dominant_cluster": 1
}
```

**Cache hit response:**

```json
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

```json
{
  "status": "cache flushed",
  "message": "All entries cleared and stats reset"
}
```

---

## Design Decisions

### Embedding & Vector Database

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim dense vectors; strong speed/quality tradeoff; CPU-friendly |
| Vector DB | ChromaDB | Persists to disk; native cosine similarity |
| Min doc length | 100 chars | Shorter docs don't embed meaningfully |
| Remove headers/footers | Yes | Reduces noise in Usenet posts |
| Retention | 16,806 / 18,846 docs | ~89% kept after cleaning |

### Fuzzy Clustering (GMM)

**Why GMM over KMeans:** GMM assigns soft cluster probabilities. Ambiguous posts (e.g. politics vs. firearms) are captured as mixed memberships rather than forced hard labels.

**Why k = 10:** BIC was lowest at k = 10 across k ∈ [5, 40]. See `backend/models/bic_curve.png`.

**PCA to 50 dimensions:** Full-covariance GMM on 384-dim vectors is intractable. PCA retains ~49.5% variance while making GMM feasible.

| Cluster | Topic | Coherence |
|---------|-------|-----------|
| 0 | Marketplace | 0.114 |
| 1 | Space & Science | 0.076 |
| 2 | Religion | 0.171 |
| 3 | Sports | 0.229 |
| 4 | Politics & Law | 0.142 |
| 5 | Hardware | 0.172 |
| 6 | Cryptography | 0.233 |
| 7 | Software | 0.131 |
| 8 | Middle East | 0.270 |
| 9 | Vehicles | 0.124 |

### Semantic Cache

- **Cluster-partitioned storage** — lookup scans only entries in the query's dominant GMM cluster (O(n/k) vs O(n))
- **Similarity** — cosine via dot product on L2-normalized embeddings
- **Thread safety** — `threading.Lock` on all cache reads/writes
- **Threshold** — **0.60** (empirically tuned; see `backend/cache/threshold_analysis.png`)

| Threshold | Hit Rate | Precision |
|-----------|----------|-----------|
| 0.50 | 0.62 | 1.00 |
| 0.65 | 0.25 | 1.00 |
| 0.80+ | 0.00 | 0.00 |

### Verified Results

```
Query 1: "What are NASA latest space missions?"    → cache_hit: false
Query 2: "Recent space exploration news from NASA" → cache_hit: true, similarity: 0.707

Cache stats:
  total_entries: 1 | hit_count: 2 | miss_count: 1 | hit_rate: 0.667
```

---

## Docker (Backend Only)

From the `backend/` directory:

```bash
docker build -t smart-semantic-search .
docker run -p 7860:7860 smart-semantic-search
```

The container listens on port **7860** (Hugging Face Spaces default). For local use, map accordingly:

```bash
docker run -p 8000:7860 smart-semantic-search
```

**Image details:**

- Base: `python:3.11-slim`
- CPU-only PyTorch (no GPU/CUDA dependencies)
- Linux-compatible `onnxruntime==1.18.1` installed separately

---

## Deploy to Hugging Face Spaces

### 1. Create a Space

Create a new Hugging Face Space with the **Docker** SDK and a blank template.

### 2. Configure files

Ensure `backend/Dockerfile` exposes port 7860:

```dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Keep the YAML frontmatter at the top of this `README.md` (required by Hugging Face).

Ensure `embeddings/chroma_db/` is **not** excluded in `.gitignore` or `.dockerignore` so the pre-built index ships with the image.

### 3. Set up Git LFS

Large model and database files should be tracked with Git LFS:

```bash
git lfs install
git lfs track "*.npy"
git lfs track "*.joblib"
git lfs track "*.sqlite3"
git lfs track "*.bin"
git lfs track "*.pickle"
```

### 4. Push to Hugging Face

```bash
git add .gitattributes
git commit -m "Setup Git LFS rules"
git add .
git commit -m "Add application code and models"
git branch -M main
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u huggingface main
```

For the frontend, deploy separately (e.g. Vercel) and set `NEXT_PUBLIC_API_URL` to your Hugging Face Space URL.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, FastAPI, Uvicorn, Sentence Transformers, ChromaDB, scikit-learn |
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS v4 |
| ML | `all-MiniLM-L6-v2`, PCA, Gaussian Mixture Models |
| Deployment | Docker, Hugging Face Spaces, Vercel (frontend) |

---

## License

See repository for license details.
