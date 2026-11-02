#!/usr/bin/env python3
"""
qdrant_rag.py

Drop-in replacement for the FAISS+scipy RAG retrieval used in
paper_vllm_confidence_interval.py and paper_experiment_judge.py.

Instead of loading a 15GB FAISS index + 1.3GB sparse.npz into RAM,
this client queries the running Qdrant server.

Usage:
    from qdrant_rag import QdrantRAG

    rag = QdrantRAG(url="http://localhost:6333")
    chunks = rag.retrieve("SAP Fiori configuration steps", top_k=15)
    # chunks -> list of str

    # Or batch:
    results = rag.retrieve_batch(["query1", "query2"], top_k=15)
    # results -> list of list of str
"""

import time
import numpy as np
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, FusionQuery, Prefetch, Fusion

COLLECTION = "sap_chunks"
BGE_M3_PATH = "/fsx/i777599/models/bge-m3/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"


class QdrantRAG:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        model_path: str = BGE_M3_PATH,
        use_fp16: bool = True,
        device: str = "cuda",
        collection: str = COLLECTION,
    ):
        print("[RAG-diag] BEFORE QdrantClient", flush=True)
        _t = time.monotonic()
        self.client = QdrantClient(url=url, timeout=60)
        print(f"[RAG-diag] AFTER QdrantClient ({time.monotonic()-_t:.3f}s)", flush=True)
        self.collection = collection

        import faulthandler
        faulthandler.dump_traceback_later(30, repeat=True)

        from FlagEmbedding import BGEM3FlagModel
        print("[RAG-diag] BEFORE BGE init", flush=True)
        _t = time.monotonic()
        # Use CPU to avoid GPU conflicts when multiple SLURM jobs share a node.
        # BGE-M3 is only used for query embedding (fast, low throughput) so CPU is fine.
        self.model = BGEM3FlagModel(model_path, use_fp16=False, devices=["cpu"], local_files_only=True)
        print(f"[RAG-diag] AFTER BGE init ({time.monotonic()-_t:.3f}s)", flush=True)

        faulthandler.cancel_dump_traceback_later()
        print("[RAG-diag] END QdrantRAG.__init__", flush=True)

    def _embed(self, texts: list[str]):
        """Returns (dense, sparse_weights) for a list of texts."""
        out = self.model.encode(
            texts,
            batch_size=32,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"], out["lexical_weights"]

    def retrieve(self, query: str, top_k: int = 15) -> list[str]:
        """Retrieve top_k chunks for a single query string."""
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int = 15) -> list[list[str]]:
        """Retrieve top_k chunks for each query. Returns list of list of str."""
        if not queries:
            return []
        dense_vecs, sparse_weights_list = self._embed(queries)

        results = []
        for i, query in enumerate(queries):
            dense_vec = dense_vecs[i].tolist()

            # Build sparse vector from lexical weights dict {token_id_str: weight}
            sp_weights = sparse_weights_list[i]
            sp_indices = [int(k) for k in sp_weights.keys()]
            sp_values  = [float(v) for v in sp_weights.values()]

            # Hybrid search: dense + sparse via RRF fusion
            hits = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=top_k * 3),
                    Prefetch(
                        query=SparseVector(indices=sp_indices, values=sp_values),
                        using="sparse",
                        limit=top_k * 3,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points

            chunks = [h.payload.get("text", "") for h in hits if h.payload.get("text")]
            results.append(chunks)

        return results


# ---- convenience: drop-in for load_rag_index usage ----

class QdrantRAGCompat:
    """
    Wraps QdrantRAG with the same interface as the old HybridRAGModel,
    so paper_vllm_confidence_interval.py works without changes beyond
    replacing load_rag_index.

    Usage in MegaScript:
        from qdrant_rag import QdrantRAGCompat
        rag_model = QdrantRAGCompat()
        # then use rag_model.retrieve(query, top_k=15) as before
    """
    def __init__(self, url: str = "http://localhost:6333", **kwargs):
        self._rag = QdrantRAG(url=url, **kwargs)

    def retrieve(self, query: str, top_k: int = 15) -> list[str]:
        return self._rag.retrieve(query, top_k=top_k)

    def retrieve_batch(self, queries: list[str], top_k: int = 15) -> list[list[str]]:
        return self._rag.retrieve_batch(queries, top_k=top_k)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Test QdrantRAG retrieval")
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--query", default="SAP Fiori configuration and setup")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    import time
    print(f"Query: {args.query!r}")
    rag = QdrantRAG(url=args.url)
    t0 = time.time()
    chunks = rag.retrieve(args.query, top_k=args.top_k)
    print(f"Retrieved {len(chunks)} chunks in {time.time()-t0:.2f}s\n")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] {c[:200]}\n")
