import json
import argparse
import time
from pathlib import Path

import numpy as np
import faiss
import torch
from tqdm import tqdm

from src.model.encoder import ByteLevelEncoder

KS = [1, 5, 10, 100]
ENCODE_BATCH_SIZE = 512


def encode_sequences(model, sequences, device, desc="Encoding"):
    """Encode a list of byte sequences. Returns float32 np.ndarray (N, D)."""
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), ENCODE_BATCH_SIZE), desc=desc):
            batch = sequences[i : i + ENCODE_BATCH_SIZE]
            max_len = max(len(s) for s in batch)
            B = len(batch)
            input_ids = torch.zeros((B, max_len), dtype=torch.long)
            attention_mask = torch.zeros((B, max_len), dtype=torch.bool)
            for j, seq in enumerate(batch):
                L = len(seq)
                if L > 0:
                    input_ids[j, :L] = torch.tensor(seq, dtype=torch.long)
                    attention_mask[j, :L] = True
            embeddings = model(input_ids.to(device), attention_mask.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def build_flat_ip(embeddings):
    D = embeddings.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(embeddings)
    return index


def build_ivf_flat(embeddings, nlist=100):
    D = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 10
    return index


def build_hnsw(embeddings, M=32):
    D = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    return index


def build_ivf_pq(embeddings, nlist=100, M=8, nbits=8):
    D = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 10
    return index


INDEX_BUILDERS = {
    "flat_ip": build_flat_ip,
    "ivf_flat": build_ivf_flat,
    "hnsw": build_hnsw,
    "ivf_pq": build_ivf_pq,
}


def run_ablation(index, query_embeddings, query_entity_ids, corpus_entity_ids):
    max_k = max(KS)
    latencies = []
    hits = {k: 0 for k in KS}
    n = len(query_embeddings)

    for i in tqdm(range(n), desc="  Querying"):
        q = query_embeddings[i : i + 1]  # (1, D)
        correct = query_entity_ids[i]

        t0 = time.perf_counter()
        _, I = index.search(q, max_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = [corpus_entity_ids[idx] for idx in I[0] if idx >= 0]
        for k in KS:
            if correct in retrieved[:k]:
                hits[k] += 1

    latencies = np.array(latencies, dtype=np.float64)
    return {
        **{f"Recall@{k}": hits[k] / n for k in KS},
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "index_size_bytes": len(faiss.serialize_index(index)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # Load test split
    records = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)
    print(f"Test records: {len(records)}")

    corpus = [{"entity_id": r["entity_id"], "bytes": r["anchor"].get("bytes", [])} for r in records]
    corpus_entity_ids = [c["entity_id"] for c in corpus]

    queries = []
    for r in records:
        for p in r["positives"]:
            queries.append({"entity_id": r["entity_id"], "bytes": p.get("bytes", [])})
    print(f"Queries: {len(queries)}")

    # Load model
    model = ByteLevelEncoder.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()
    hidden_dim = model.config.hidden_dim

    # Encode corpus and queries once
    corpus_embeddings = encode_sequences(model, [c["bytes"] for c in corpus], device, desc="Encoding corpus")
    query_embeddings = encode_sequences(model, [q["bytes"] for q in queries], device, desc="Encoding queries")
    query_entity_ids = [q["entity_id"] for q in queries]

    # Run ablation for each index type
    results = {}
    for name, builder in INDEX_BUILDERS.items():
        print(f"\nBuilding {name} index...", end=" ", flush=True)
        t0 = time.perf_counter()
        index = builder(corpus_embeddings)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

        results[name] = run_ablation(index, query_embeddings, query_entity_ids, corpus_entity_ids)

        r10 = results[name]["Recall@10"]
        lat = results[name]["latency_ms"]["mean"]
        print(f"  Recall@10={r10:.3f}  mean_latency={lat:.2f}ms")

    output = {
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "hidden_dim": hidden_dim,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Summary table
    print(f"\n{'Index':<12} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@100':>7} {'lat_mean':>10} {'lat_p99':>10} {'size_MB':>9}")
    for name, res in results.items():
        print(
            f"{name:<12} "
            f"{res['Recall@1']:>6.3f} "
            f"{res['Recall@5']:>6.3f} "
            f"{res['Recall@10']:>6.3f} "
            f"{res['Recall@100']:>7.3f} "
            f"{res['latency_ms']['mean']:>10.2f} "
            f"{res['latency_ms']['p99']:>10.2f} "
            f"{res['index_size_bytes']/1e6:>9.1f}"
        )


if __name__ == "__main__":
    main()
