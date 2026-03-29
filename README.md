# Enterprise RAG Knowledge Search System

A production-style starter for Retrieval-Augmented Generation (RAG) over internal enterprise knowledge.

## Stack
- Python + FastAPI API layer
- SQLAlchemy + SQLite metadata store (users, docs, permissions, chunks, audit logs)
- Semantic vector retrieval with batched embeddings
- RBAC + document-level permissions
- Dockerized API + indexing pipeline with service health checks

## Features
- Document ingestion and chunking
- Embeddings + cosine similarity retrieval
- Optional C++ SIMD + multi-threaded vector ranking extension (cosine + euclidean)
- Cited answers with chunk-level traceability
- Role-based access (`admin`, `analyst`, `viewer`)
- Fine-grained document sharing via explicit permissions
- Search/index audit logs with latency tracking
- Evaluation script for recall@k quality checks
- Query embedding cache + batched chunk embeddings

## Quickstart

1. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

2. Build the native vector engine (optional but recommended):
```bash
make build-cpp
```

3. Run API:
```bash
python3 -m uvicorn app.main:app --reload
```

4. Open demo web app:
```text
http://127.0.0.1:8000/demo
```

5. Index sample docs:
```bash
./scripts/index_sample_docs.sh
```

6. Search:
```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -H 'X-User: victor_viewer' \
  -d '{"query":"What are required SSO controls?","top_k":3}'
```

## Demo Web App
Open `http://127.0.0.1:8000/demo` to use the demo console.

What it does:
- Lets you switch between demo users (`admin`, `analyst`, `viewer`) to show role-based access behavior.
- Lets you index local text files into the RAG system (`private` or `public` visibility).
- Shows which documents are currently accessible to the selected user.
- Runs semantic search and returns a generated answer with chunk-level citations.
- Shows a request log so audiences can follow each API action during the demo.

## Suggested Demo Questions
Use these questions after indexing the sample docs:
- `What are required SSO controls?`
- `Is MFA required for remote sessions?`
- `What is needed for production database access?`
- `What approvals are needed for payroll changes?`
- `What is the payroll cutoff time?`
- `How long must support tickets be retained?`
- `What happens when there is a legal hold notice?`
- `How should archived PII be protected?`
- `How are retention exceptions verified?`

Note: Off-topic questions like `What is SQL?` are not covered by the sample docs and may return weak or no relevant context.

## Demo Users
- `alice_admin` (`admin`)
- `amy_analyst` (`analyst`)
- `victor_viewer` (`viewer`)

Pass user identity in `X-User` header.

## API Endpoints
- `GET /` (web demo UI)
- `GET /demo` (web demo UI)
- `GET /health`
- `GET /users/me`
- `POST /documents/index`
- `GET /documents`
- `POST /documents/{document_id}/permissions`
- `GET /documents/{document_id}/can_read`
- `POST /search`

## Retrieval Evaluation
```bash
python3 scripts/evaluate_retrieval.py --username victor_viewer --top-k 5
```

## SIMD Vector Search Benchmark
Run latency benchmarks for the C++ engine:
```bash
python3 scripts/benchmark_vector_engine.py --num-vectors 200000 --dim 384 --top-k 10 --num-queries 20 --compare-numpy
```

Stress-test with 1M embeddings (requires ~1.5GB+ RAM):
```bash
python3 scripts/benchmark_vector_engine.py --num-vectors 1000000 --dim 384 --top-k 10 --num-queries 20 --compare-numpy
```

Or via Make:
```bash
make bench-cpp
```

## Smoke Test
Run an end-to-end validation (start API, check health/auth, index docs, search, and eval):
```bash
bash scripts/smoke_test.sh
```

Or via Make:
```bash
make smoke
```

Optional flags:
```bash
BASE_URL=http://127.0.0.1:8000 START_SERVER=0 bash scripts/smoke_test.sh
```

## Docker
```bash
docker compose up --build
```
