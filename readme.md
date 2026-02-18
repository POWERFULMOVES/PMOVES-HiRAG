# PMOVES-HiRAG

Hybrid Retrieval-Augmented Generation (RAG) system combining vector search, graph search, and full-text search with hierarchical knowledge extraction and cross-encoder reranking.

This is PMOVES.AI's integration and deployment of [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/abs/2503.10150), accepted to EMNLP 2025 Findings.

## Overview

PMOVES-HiRAG is a production-ready implementation of the HiRAG research system, designed for high-performance knowledge retrieval in the PMOVES.AI ecosystem. It features:

- **Hierarchical Knowledge Graph**: Multi-level entity extraction with GMM clustering
- **Hybrid Retrieval**: Combines three search modalities:
  - Vector search via Qdrant (semantic similarity)
  - Graph search via Neo4j (relationship traversal)
  - Full-text search via Meilisearch (keyword matching)
- **Advanced Reranking**: Cross-encoder reranking with Qwen3-Reranker-4B or BGE-Reranker
- **Multiple Query Modes**: Local, global, bridge, and naive RAG strategies
- **GPU Acceleration**: Optional GPU support for embedding and reranking

## Architecture

![HiRAG Architecture](./imgs/hirag_ds_trans.drawio.png)

### Key Components

1. **Text Chunking**: Token-based chunking with configurable overlap
2. **Entity Extraction**: LLM-powered hierarchical entity extraction
3. **GMM Clustering**: Gaussian Mixture Model for semantic clustering
4. **Cross-Encoder Reranking**: Final relevance scoring and ranking
5. **Multi-Modal Integration**: Supports text, and optionally image/audio geometry

## Quick Start

### Installation

```bash
# Clone the repository
cd PMOVES-HiRAG
pip install -e .
```

### Basic Usage

```python
from hirag import HiRAG, QueryParam

# Initialize HiRAG
graph_func = HiRAG(
    working_dir="./hirag_cache",
    enable_llm_cache=True,
    enable_hierachical_mode=True,
    embedding_batch_num=6,
    embedding_func_max_async=8,
    enable_naive_rag=True
)

# Index documents
with open("path_to_your_context.txt", "r") as f:
    graph_func.insert(f.read())

# Query with hierarchical mode
result = graph_func.query(
    "What is the main topic?",
    param=QueryParam(mode="hi")
)
print(result)
```

### Query Modes

- `mode="hi"` - Full HiRAG with local, global, and bridge knowledge
- `mode="naive"` - Traditional vector-only RAG
- `mode="hi_local"` - Local knowledge only (entity-level)
- `mode="hi_global"` - Global knowledge only (community-level)
- `mode="hi_bridge"` - Bridge knowledge only (inter-cluster)
- `mode="hi_nobridge"` - HiRAG without bridge entities

## PMOVES.AI Integration

### Service Architecture

PMOVES-HiRAG is deployed as the **Hi-RAG Gateway v2** service within PMOVES.AI:

- **CPU Service**: `hi-rag-gateway-v2` (Port 8086)
- **GPU Service**: `hi-rag-gateway-v2-gpu` (Port 8087)

### API Endpoints

#### Primary Endpoint: Hybrid Query

```bash
POST http://localhost:8086/hirag/query
Content-Type: application/json

{
  "query": "What are the key concepts?",
  "top_k": 10,
  "rerank": true,
  "alpha": 0.7,
  "namespace": "pmoves"
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Retrieved text chunk",
      "score": 0.95,
      "source": "document_id",
      "metadata": {}
    }
  ],
  "metrics": {
    "vector_count": 50,
    "graph_count": 20,
    "fulltext_count": 30,
    "reranked": true,
    "total_ms": 250
  }
}
```

#### Health & Status

```bash
GET http://localhost:8086/healthz
GET http://localhost:8086/hirag/admin/stats
```

### Environment Variables

#### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant vector database URL |
| `QDRANT_COLLECTION` | `pmoves_chunks_qwen3` | Qdrant collection name |
| `SENTENCE_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `INDEXER_NAMESPACE` | `pmoves` | Default namespace for queries |

#### Retrieval Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPHA` | `0.7` | Vector/text search balance (0-1) |
| `GRAPH_BOOST` | `0.15` | Graph result boost factor |
| `ENTITY_CACHE_TTL` | `60` | Entity cache TTL (seconds) |
| `ENTITY_CACHE_MAX` | `1000` | Max cached entities |

#### Reranking Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANK_ENABLE` | `true` | Enable cross-encoder reranking |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Reranker model ID |
| `RERANK_TOPN` | `50` | Candidates for reranking |
| `RERANK_K` | `10` | Final results after reranking |
| `RERANK_FUSION` | `mul` | Score fusion method (mul/wsum) |
| `RERANK_PROVIDER` | `flag` | Reranker backend (flag/tensorzero) |

#### Search Engine Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MEILI` | `true` | Enable Meilisearch |
| `MEILI_URL` | `http://meilisearch:7700` | Meilisearch endpoint |
| `MEILI_API_KEY` | - | Meilisearch API key |
| `NEO4J_URI` | - | Neo4j graph database URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | - | Neo4j password |

#### Optional: TensorZero Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSORZERO_BASE_URL` | `http://tensorzero-gateway:3000` | TensorZero gateway URL |
| `TENSORZERO_API_KEY` | - | TensorZero API key |
| `TENSORZERO_EMBED_MODEL` | `tensorzero::embedding_model_name::qwen3_embedding_4b_local` | Embedding model via TensorZero |

#### Optional: Ollama Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_OLLAMA_EMBED` | `false` | Use Ollama for embeddings |
| `OLLAMA_URL` | `http://pmoves-ollama:11434` | Ollama endpoint |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:4b` | Ollama embedding model |

## Docker Deployment

### CPU Service

```bash
docker compose --profile workers up -d hi-rag-gateway-v2
```

### GPU Service

```bash
docker compose --profile gpu up -d hi-rag-gateway-v2-gpu
```

### Both Services

```bash
make up-both-gateways
```

## Dependencies

### Python Requirements

Core dependencies (see `requirements.txt` for full list):

- `nano_vectordb==0.0.2` - Lightweight vector database
- `neo4j==5.25.0` - Graph database client
- `networkx==3.3` - Graph algorithms
- `openai==1.61.1` - LLM integration
- `pydantic==2.10.6` - Data validation
- `scikit_learn==1.6.1` - ML algorithms (GMM)
- `sentence_transformers` - Embedding models
- `FlagEmbedding` - Reranker models
- `transformers==4.47.1` - HuggingFace models
- `umap_learn==0.5.6` - Dimensionality reduction

### External Services

Required for PMOVES.AI integration:

- **Qdrant** (Port 6333): Vector embeddings storage
- **Neo4j** (Ports 7474/7687): Knowledge graph
- **Meilisearch** (Port 7700): Full-text search
- **Optional: TensorZero** (Port 3030): LLM gateway & observability
- **Optional: Ollama** (Port 11434): Local model inference

## Configuration Files

### config.yaml

Example configuration for standalone usage:

```yaml
# OpenAI Configuration
openai:
  embedding_model: "text-embedding-ada-002"
  model: "gpt-4o"
  api_key: "your-key"
  base_url: "https://api.openai.com/v1"

# Model Parameters
model_params:
  openai_embedding_dim: 1536
  max_token_size: 8192

# HiRAG Configuration
hirag:
  working_dir: "hirag_cache"
  enable_llm_cache: true
  enable_hierachical_mode: true
  embedding_batch_num: 6
  embedding_func_max_async: 8
  enable_naive_rag: true
```

## Advanced Usage

### Using Third-Party LLM Providers

HiRAG supports multiple LLM backends. See example scripts:

- `hi_Search_openai.py` - OpenAI integration
- `hi_Search_deepseek.py` - DeepSeek integration
- `hi_Search_glm.py` - ChatGLM integration

### Evaluation & Benchmarking

Run evaluations on standard datasets:

```bash
cd eval

# Extract context from datasets
python extract_context.py -i ./datasets/mix -o ./datasets/mix

# Insert context to graph
python insert_context_deepseek.py

# Test with different modes
python test_deepseek.py -d mix -m hi
python test_deepseek.py -d mix -m naive
python test_deepseek.py -d mix -m hi_nobridge

# Batch evaluation
python batch_eval.py -m request -api openai
python batch_eval.py -m result -api openai
```

## Performance Benchmarks

HiRAG significantly outperforms traditional RAG and other graph-based approaches across multiple domains:

| Comparison | Dataset | HiRAG Win Rate |
|------------|---------|----------------|
| vs Naive RAG | Mix | 87.6% |
| vs GraphRAG | Mix | 64.1% |
| vs LightRAG | Mix | 65.9% |
| vs FastGraphRAG | Mix | 99.2% |
| vs KAG | Mix | 97.7% |

See the [research paper](https://arxiv.org/abs/2503.10150) for detailed results.

## Integration Patterns

### Query from External Services

```python
import httpx

# Query HiRAG from any PMOVES service
response = httpx.post(
    "http://hi-rag-gateway-v2:8086/hirag/query",
    json={
        "query": "What is Claude Code?",
        "top_k": 5,
        "rerank": True,
        "namespace": "pmoves"
    }
)
results = response.json()["results"]
```

### Use with SupaSerch

PMOVES-HiRAG is integrated with SupaSerch for deep research orchestration:

```bash
# SupaSerch automatically queries HiRAG
curl -X POST http://localhost:8099/supaserch/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain multi-agent systems", "depth": "deep"}'
```

### Use with Agent Zero

Agent Zero MCP interface can leverage HiRAG:

```bash
# Agent Zero calls HiRAG for knowledge retrieval
curl -X POST http://localhost:8080/mcp/command \
  -H "Content-Type: application/json" \
  -d '{"command": "knowledge_search", "query": "database design patterns"}'
```

## Development

### Running Tests

```bash
# Functional tests
cd /path/to/pmoves
bash tests/functional/test_hirag_query.sh

# Integration tests
pytest tests/test_hirag_gateway.py
```

### Building Images

```bash
# CPU image
docker build -f services/hi-rag-gateway-v2/Dockerfile -t pmoves-hirag-v2:latest .

# GPU image
docker build -f services/hi-rag-gateway-v2/Dockerfile.gpu -t pmoves-hirag-v2-gpu:latest .
```

## Monitoring & Observability

### Prometheus Metrics

All HiRAG gateway instances expose `/metrics` endpoint:

```bash
curl http://localhost:8086/metrics
```

Key metrics:
- Request count and latency
- Cache hit rates
- Reranking performance
- Storage backend health

### Grafana Dashboard

PMOVES.AI includes a pre-configured Grafana dashboard at `http://localhost:3000` showing:
- Query throughput
- Response time percentiles
- Search modality breakdown
- Error rates

### Loki Logs

Centralized logging via Loki/Promtail:

```bash
# View HiRAG logs
curl -G http://localhost:3100/loki/api/v1/query \
  --data-urlencode 'query={service="hi-rag-gateway-v2"}'
```

## Troubleshooting

### Reranker Model Loading Issues

If you see reranker initialization errors:

1. Check GPU availability: `docker exec hi-rag-gateway-v2-gpu nvidia-smi`
2. Verify model path: `RERANK_MODEL_PATH` points to valid snapshot
3. Check memory: Qwen3-Reranker-4B requires ~8GB VRAM
4. Fallback to CPU: Use `hi-rag-gateway-v2` service instead

### Neo4j Connection Failures

```bash
# Verify Neo4j is running
docker ps | grep neo4j

# Check credentials
echo $NEO4J_PASSWORD

# Test connection
docker exec hi-rag-gateway-v2 python -c "from neo4j import GraphDatabase; GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'password')).verify_connectivity()"
```

### Qdrant Collection Not Found

```bash
# List collections
curl http://localhost:6333/collections

# Create collection (if needed)
curl -X PUT http://localhost:6333/collections/pmoves_chunks_qwen3 \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 384, "distance": "Cosine"}}'
```

## Documentation

- **Research Paper**: [arXiv:2503.10150](https://arxiv.org/abs/2503.10150)
- **Dataset**: [UltraDomain on HuggingFace](https://huggingface.co/datasets/TommyChien/UltraDomain)
- **Original Implementation**: [hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG)
- **PMOVES Services**: See `.claude/context/services-catalog.md` in PMOVES.AI repo

## Contributing

Contributions to PMOVES-HiRAG are welcome! Please follow the PMOVES.AI contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest` and `make verify-all`
5. Submit a pull request to `main` branch

## Acknowledgments

This project builds upon:
- [HiRAG](https://github.com/hhy-huang/HiRAG) by Haoyu Huang et al. (EMNLP 2025)
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag) - GraphRAG implementation
- [RAPTOR](https://github.com/parthsarthi03/raptor) - Recursive tree structure approach

## Citation

If you use PMOVES-HiRAG in your research, please cite:

```bibtex
@article{huang2025retrieval,
  title={Retrieval-Augmented Generation with Hierarchical Knowledge},
  author={Huang, Haoyu and Huang, Yongfeng and Yang, Junjie and Pan, Zhenyu and Chen, Yongqiang and Ma, Kaili and Chen, Hongzhi and Cheng, James},
  journal={arXiv preprint arXiv:2503.10150},
  year={2025}
}
```

## License

This project inherits the MIT License from the original HiRAG implementation. See [LICENSE](./LICENSE) for details.

## Support

For issues and questions:
- PMOVES.AI integration: Open an issue in the PMOVES.AI repository
- HiRAG core functionality: Refer to [original HiRAG repo](https://github.com/hhy-huang/HiRAG)
- Production deployment: See PMOVES.AI documentation at `.claude/CLAUDE.md`
