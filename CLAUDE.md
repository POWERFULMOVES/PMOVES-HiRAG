# PMOVES-HiRAG Developer Context

**Always-on context for Claude Code CLI when working in the PMOVES-HiRAG repository.**

## Architecture Overview

PMOVES-HiRAG is a **hybrid RAG (Retrieval-Augmented Generation) gateway** combining three retrieval backends with cross-encoder reranking:
- **Qdrant** - Vector similarity search (dense embeddings)
- **Neo4j** - Knowledge graph traversal (entity relationships)
- **Meilisearch** - Full-text keyword search (typo-tolerant)

Results from all three backends are fused and reranked using a cross-encoder model for maximum relevance.

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `api/main.py` | `api/` | FastAPI gateway, route handlers |
| `api/rag_engine.py` | `api/` | Core RAG orchestration logic |
| `api/retriever_qdrant.py` | `api/` | Qdrant vector retrieval |
| `api/retriever_neo4j.py` | `api/` | Neo4j graph retrieval |
| `api/retriever_meilisearch.py` | `api/` | Meilisearch full-text retrieval |
| `api/reranker.py` | `api/` | Cross-encoder reranking (Qwen3-Reranker) |
| `api/gdb_neo4j.py` | `api/` | Neo4j graph database operations |
| `api/embedding_provider.py` | `api/` | Embedding generation (multi-provider) |

## Security Posture

- **P1 FIXED (Phase H 2026-02-17):** Cypher injection — f-string label construction replaced with strict allowlist validation
- **P1 FIXED:** Default credentials removed from committed configuration
- **P2 OPEN:** `env.shared` uses `export` syntax (Docker `env_file` incompatible)
- **GREEN:** Cross-encoder reranking uses parameterized queries
- **GREEN:** Input validation on query parameters

## APIs

### Gateway (Port 8086 CPU / 8087 GPU)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/healthz` | GET | Health check with backend connectivity |
| `/hirag/query` | POST | Hybrid RAG query with optional reranking |
| `/hirag/index` | POST | Index new content across all backends |

### Query Request Format

```json
{
  "query": "your question here",
  "top_k": 10,
  "rerank": true,
  "mode": "global",
  "collection": "pmoves_chunks_qwen3"
}
```

### Query Modes

| Mode | Description |
|------|-------------|
| `local` | Single-backend retrieval (fastest) |
| `global` | All backends + fusion + reranking (best quality) |
| `bridge` | Graph-guided vector search (relationship-aware) |
| `naive` | Simple vector similarity only (baseline) |

## Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `QDRANT_URL` | Qdrant vector store URL | `http://qdrant:6333` |
| `QDRANT_COLLECTION` | Vector collection name | `pmoves_chunks_qwen3` |
| `NEO4J_URI` | Neo4j bolt connection | `bolt://neo4j:7687` |
| `NEO4J_AUTH` | Neo4j credentials | Required |
| `MEILI_URL` | Meilisearch URL | `http://meilisearch:7700` |
| `MEILI_MASTER_KEY` | Meilisearch auth key | Required |
| `RERANK_ENABLE` | Enable cross-encoder reranking | `true` |
| `RERANK_MODEL` | Reranker model ID | `Qwen/Qwen3-Reranker-4B` |
| `RERANK_MODEL_PATH` | Local model path | `/models/qwen/Qwen3-Reranker-4B` |
| `RERANK_TOPN` | Candidates before reranking | `50` |
| `RERANK_K` | Final top-K after reranking | `10` |
| `RERANK_FUSION` | Fusion method (`mul`/`add`) | `mul` |
| `SENTENCE_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `NATS_URL` | NATS connection URL | `nats://nats:pmoves@nats:4222` |

## Development

### Local Setup

```bash
cd PMOVES-HiRAG
pip install -r requirements.txt

# CPU mode
uvicorn api.main:app --host 0.0.0.0 --port 8086

# GPU mode (with reranker)
RERANK_ENABLE=true uvicorn api.main:app --host 0.0.0.0 --port 8087
```

### Testing

```bash
# Health check
curl http://localhost:8086/healthz

# Query
curl -X POST http://localhost:8086/hirag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 5, "rerank": false}'
```

## PMOVES.AI Integration

### Docked Mode

- **Compose profiles:** `workers` (CPU), `gpu` (GPU with reranker)
- **Ports:** 8086 (CPU), 8087 (GPU)
- **Networks:** `pmoves-net`, `data-net`
- **Depends on:** Qdrant, Neo4j, Meilisearch (all `service_healthy`)

### No Direct NATS

Hi-RAG is a **request/reply service** — it does not publish or subscribe to NATS subjects. Other services call it via HTTP.

### Consumers

Services that query Hi-RAG:
- **SupaSerch** - Multi-source research
- **DeepResearch** - Research planning
- **Agent Zero** - Knowledge retrieval tool
- **Extract Worker** - Content indexing

## Common Gotchas

1. **Reranker GPU memory:** Qwen3-Reranker-4B requires ~8GB VRAM; falls back to CPU if unavailable
2. **Sentence cache:** First query pre-loads sentence transformer; subsequent queries are fast
3. **Neo4j labels:** Must use strict allowlist validation (Phase H Cypher injection fix)
4. **Collection mismatch:** Ensure `QDRANT_COLLECTION` matches the collection used by Extract Worker
5. **Embedding dimension:** `all-MiniLM-L6-v2` produces 384-dim vectors; `qwen3-embedding:4b` produces 2048-dim

<!-- PMOVES.AI-CONTEXT-TAGS -->
## PMOVES.AI Skill Hints

**Primary Skills:** `/search:hirag`, `/deploy:up`, `/health:quick`, `/gpu:status`
**Context Files:** `services-catalog.md`
**Domain Tags:** `retrieval`, `knowledge`
**Context Tier:** 2 (On-Demand (Major Subsystem))
<!-- /PMOVES.AI-CONTEXT-TAGS -->
