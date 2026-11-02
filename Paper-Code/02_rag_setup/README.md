# 02_rag_setup — RAG Pipeline

Builds and serves the Qdrant hybrid-search index (dense + sparse, BGE-M3) used as the RAG
baseline in the SAE evaluation pipeline.

Two parallel pipelines exist: **SAP** (domain corpus) and **Fintrain** (Salesforce/FinTrain).

---

## Pipeline Overview

### SAP

```
1. sbatch create_chunks_sap.sh        # split SAP JSONL docs → 9 shards
2. sbatch embed_shards_sap.sh         # SLURM array: embed each shard (BGE-M3, dense+sparse)
3. sbatch merge_vector_db_sap.sh      # merge partials → vector_db_final/merged/
4. sbatch qdrant_server_sap.sh        # start Qdrant Docker (ports 6333/6334, persistent)
5. sbatch build_qdrant_fast_sap.sh    # upload FAISS+sparse → Qdrant collection: sap_chunks
```

### Fintrain

```
1. sbatch create_chunks_fintrain.sh   # stream Salesforce/FinTrain → shards
2. sbatch embed_shards_fintrain.sh    # SLURM array: embed each shard
3. sbatch merge_vector_db_fintrain.sh # merge partials → fintrain_vector_db/merged/
4. sbatch qdrant_server_fintrain.sh   # start Qdrant Docker (ports 6335/6336, persistent)
5. sbatch build_qdrant_fast_fintrain.sh # upload → Qdrant collection: fintrain_chunks
```

---

## Data Paths

| What | Path |
|---|---|
| SAP shards | `/fsx/i777599/cache/vecdb_shards/` |
| SAP merged index | `/shared/home/i777599/nova_model/SAE/data/RAG/vector_db_final/merged/` |
| SAP Qdrant storage | `/shared/home/i777599/nova_model/SAE/data/qdrant_storage/` |
| Fintrain shards | `/fsx/i777599/cache/fintrain_shards/` |
| Fintrain merged index | `/shared/home/i777599/nova_model/SAE/data/RAG/fintrain_vector_db/merged/` |
| Fintrain Qdrant storage | `/shared/home/i777599/nova_model/SAE/data/qdrant_fintrain_storage/` |

---

## Qdrant Servers (currently running)

| Collection | Node | REST port | gRPC port |
|---|---|---|---|
| `sap_chunks` | cpu-dy-c59xlarge-1 | 6333 | 6334 |
| `fintrain_chunks` | cpu-dy-c59xlarge-2 | 6335 | 6336 |

To query Qdrant from Python use `qdrant_rag.py` (in `util/`):
```python
from qdrant_rag import QdrantRAG
rag = QdrantRAG(url="http://cpu-dy-c59xlarge-1:6333")
chunks = rag.retrieve("SAP Fiori configuration", top_k=15)
```

---

## File Reference

| File | Role |
|---|---|
| `create_chunks_sap.py` | Reads SAP JSONL files, chunks text, writes shards |
| `create_chunks_sap.sh` | SLURM wrapper for above |
| `create_chunks_fintrain.py` | Streams Salesforce/FinTrain HuggingFace dataset, writes shards |
| `create_chunks_fintrain.sh` | SLURM wrapper for above |
| `embed_shard.py` | Embeds one shard with BGE-M3 (dense + sparse), writes partial index |
| `embed_shards_sap.sh` | SLURM array job (9 tasks) — one per SAP shard |
| `embed_shards_fintrain.sh` | SLURM array job — one per Fintrain shard |
| `merge_vector_db.py` | Merges all partial indexes → single `index.faiss` + `sparse.npz` |
| `merge_vector_db_sap.sh` | SLURM wrapper for SAP merge |
| `merge_vector_db_fintrain.sh` | SLURM wrapper for Fintrain merge |
| `qdrant_server_sap.sh` | Starts Qdrant Docker for SAP (ports 6333/6334) |
| `qdrant_server_fintrain.sh` | Starts Qdrant Docker for Fintrain (ports 6335/6336) |
| `build_qdrant_index_fast.py` | Uploads FAISS+sparse index into Qdrant via gRPC (fast, parallel) |
| `build_qdrant_fast_sap.sh` | SLURM wrapper — uploads SAP index to `sap_chunks` collection |
| `build_qdrant_fast_fintrain.sh` | SLURM wrapper — uploads Fintrain index to `fintrain_chunks` collection |
