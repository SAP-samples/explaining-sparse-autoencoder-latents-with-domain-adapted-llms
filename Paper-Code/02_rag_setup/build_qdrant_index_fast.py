#!/usr/bin/env python3
"""
build_qdrant_index_fast.py

Fast version of build_qdrant_index.py:
  - Parallel upload threads (--num-workers)
  - Vectorized sparse batch extraction (no per-row Python loop)
  - Larger default batch size (4096)
  - Pre-converts CSR batch to COO once, then splits by row

Usage:
    python build_qdrant_index_fast.py \
        --index-dir /path/to/vector_db_final/merged \
        --qdrant-url http://cpu-dy-c59xlarge-1:6333 \
        [--batch-size 4096] \
        [--num-workers 8] \
        [--start-from 0] \
        [--collection sap_chunks]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import scipy.sparse as sp
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DENSE_DIM  = 1024
SPARSE_DIM = 250002


def create_collection(client, collection, recreate=False):
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if recreate:
            print(f"Deleting existing collection '{collection}'...")
            client.delete_collection(collection)
        else:
            print(f"Collection '{collection}' already exists — resuming upload.")
            return
    print(f"Creating collection '{collection}'...")
    from qdrant_client.models import OptimizersConfigDiff, HnswConfigDiff
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=100000,
            max_optimization_threads=16,
            max_segment_size=500000,
            default_segment_number=4,
        ),
        hnsw_config=HnswConfigDiff(
            max_indexing_threads=16,
        ),
    )
    print("Collection created.")


def load_sources(index_dir):
    meta_path = f"{index_dir}/metadata.jsonl"
    print("Loading metadata...")
    metadata = []
    with open(meta_path) as f:
        for line in f:
            metadata.append(json.loads(line))
    print(f"  {len(metadata)} metadata entries loaded.")

    source_lines = defaultdict(list)
    for doc_id, m in enumerate(metadata):
        source_lines[m["source"]].append((m["line"], doc_id))

    print("Preloading chunk texts...")
    chunk_texts = {}
    for source, line_doc_pairs in source_lines.items():
        needed = defaultdict(list)
        for line, doc_id in line_doc_pairs:
            needed[line].append(doc_id)
        try:
            with open(source) as f:
                for li, raw in enumerate(f):
                    if li in needed:
                        try:
                            text = json.loads(raw).get("text", "").strip()
                            for doc_id in needed[li]:
                                chunk_texts[doc_id] = text[:1000]
                        except (json.JSONDecodeError, AttributeError):
                            for doc_id in needed[li]:
                                chunk_texts[doc_id] = ""
        except OSError:
            continue
    print(f"  {len(chunk_texts)} texts loaded.")
    return metadata, chunk_texts


def build_points(ids, dense_vecs, sp_batch_csr, metadata, chunk_texts):
    """Build PointStructs for a batch. Vectorized sparse extraction."""
    # Convert entire CSR batch to COO once — much faster than per-row getrow()
    coo = sp_batch_csr.tocoo()
    # Group COO entries by row
    row_indices = defaultdict(list)
    row_values  = defaultdict(list)
    for r, c, v in zip(coo.row, coo.col, coo.data):
        row_indices[r].append(int(c))
        row_values[r].append(float(v))

    points = []
    for i, doc_id in enumerate(ids):
        m = metadata[doc_id]
        points.append(PointStruct(
            id=doc_id,
            vector={
                "dense":  dense_vecs[i].tolist(),
                "sparse": SparseVector(
                    indices=row_indices.get(i, []),
                    values=row_values.get(i, []),
                ),
            },
            payload={
                "source": m.get("source", ""),
                "line":   m.get("line", 0),
                "text":   chunk_texts.get(doc_id, ""),
            },
        ))
    return points


def upload(client, collection, index_dir, batch_size, start_from, num_workers):
    t_load = time.time()
    print("Loading FAISS index...", flush=True)
    faiss_index = faiss.read_index(f"{index_dir}/index.faiss")
    total = faiss_index.ntotal
    print(f"  {total} vectors, dim={faiss_index.d} ({time.time()-t_load:.1f}s)", flush=True)

    t_sp = time.time()
    print("Loading sparse matrix...", flush=True)
    sparse_mat = sp.load_npz(f"{index_dir}/sparse.npz")
    print(f"  shape={sparse_mat.shape}, nnz={sparse_mat.nnz} ({time.time()-t_sp:.1f}s)", flush=True)

    t_meta = time.time()
    metadata, chunk_texts = load_sources(index_dir)
    print(f"  metadata+texts loaded ({time.time()-t_meta:.1f}s)", flush=True)

    sample = faiss_index.reconstruct_batch(
        list(range(min(10, total))),
        np.zeros((min(10, total), faiss_index.d), dtype=np.float32)
    )
    norms = np.linalg.norm(sample, axis=1)
    print(f"  Dense vectors already L2-normalized: {np.allclose(norms, 1.0, atol=0.01)}", flush=True)

    print(f"\nUploading {total} points to Qdrant", flush=True)
    print(f"  batch={batch_size}, workers={num_workers}, start={start_from}", flush=True)
    batches = list(range(start_from, total, batch_size))

    def upload_batch(batch_start):
        batch_end = min(batch_start + batch_size, total)
        ids = list(range(batch_start, batch_end))
        n = len(ids)
        dense_vecs = np.zeros((n, DENSE_DIM), dtype=np.float32)
        faiss_index.reconstruct_batch(ids, dense_vecs)
        sp_batch = sparse_mat[batch_start:batch_end].tocsr()
        points = build_points(ids, dense_vecs, sp_batch, metadata, chunk_texts)
        client.upsert(collection_name=collection, points=points, wait=False)
        return batch_end

    t0 = time.time()

    t0 = time.time()
    uploaded = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending = {}
        batch_iter = iter(batches)
        for b in batch_iter:
            if len(pending) >= num_workers * 2:
                break
            pending[pool.submit(upload_batch, b)] = b

        while pending:
            done_fut = next(as_completed(pending))
            batch_end = done_fut.result()
            del pending[done_fut]
            uploaded += batch_size
            elapsed = time.time() - t0
            rate = uploaded / elapsed if elapsed > 0 else 0
            remaining = (total - uploaded - start_from) / rate if rate > 0 else 0
            print(f"  {uploaded+start_from}/{total} ({100*(uploaded+start_from)/total:.1f}%) "
                  f"| {rate:.0f} pts/s | ETA {remaining/60:.1f} min", flush=True)
            b = next(batch_iter, None)
            if b is not None:
                pending[pool.submit(upload_batch, b)] = b

    upload_time = time.time() - t0
    print(f"\nAll batches sent in {upload_time/3600:.2f}h. Waiting for Qdrant to finish indexing...")

    # wait=False means Qdrant indexes asynchronously — poll until points_count == total
    import urllib.request as _ur, json as _json
    qdrant_rest_url = f"http://{args.qdrant_host}:6333" if hasattr(args, 'qdrant_host') else "http://localhost:6333"
    poll_interval = 15
    while True:
        try:
            resp = _ur.urlopen(f"{qdrant_rest_url}/collections/{collection}", timeout=10)
            info = _json.loads(resp.read())["result"]
            pts = info["points_count"]
            queue = info.get("update_queue", {}).get("length", "?")
            status = info["status"]
            print(f"  indexed: {pts:,} / {total:,}  queue={queue}  status={status}", flush=True)
            if pts >= total:
                break
        except Exception as e:
            print(f"  poll error: {e}", flush=True)
        time.sleep(poll_interval)

    print(f"\nDone. Total time: {(time.time()-t0)/3600:.2f}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir",   default="/shared/home/i777599/nova_model/SAE/data/RAG/vector_db_final/merged")
    ap.add_argument("--qdrant-url",  default="http://localhost:6333")
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--grpc-port",   type=int, default=6334)
    ap.add_argument("--collection",  default="sap_chunks")
    ap.add_argument("--batch-size",  type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--start-from",  type=int, default=0)
    ap.add_argument("--recreate",    action="store_true")
    args = ap.parse_args()

    client = QdrantClient(host=args.qdrant_host, grpc_port=args.grpc_port, prefer_grpc=True, timeout=3600)
    print(f"Connected to Qdrant via gRPC at {args.qdrant_host}:{args.grpc_port}")
    print(f"Collection: {args.collection}")

    create_collection(client, args.collection, recreate=args.recreate)
    upload(client, args.collection, args.index_dir, args.batch_size, args.start_from, args.num_workers)

    info = client.get_collection(args.collection)
    print(f"\nCollection '{args.collection}': {info.points_count} points indexed.")


if __name__ == "__main__":
    main()
