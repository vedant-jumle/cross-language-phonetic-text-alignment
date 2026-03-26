import numpy as np
import argparse
import json
import os
from tqdm import tqdm
from pathlib import Path

PARALLEL_RETRIEVERS = {"levenshtein", "bm25", "soundex", "transliterate"}

def _run_query(args):
    q, index, max_k, retrieve_fn = args
    retrieved = retrieve_fn(q["text"], q["bytes"], index, max_k)
    rank = None
    for i, eid in enumerate(retrieved):
        if eid == q["entity_id"]:
            rank = i + 1
            break
    return rank, q["type"], q.get("script", "latin")

def _run_query_batch(args):
    queries_chunk, index, max_k, retrieve_fn = args
    results = []
    for q in queries_chunk:
        retrieved = retrieve_fn(q["text"], q["bytes"], index, max_k)
        rank = None
        for i, eid in enumerate(retrieved):
            if eid == q["entity_id"]:
                rank = i + 1
                break
        results.append((rank, q["type"], q.get("script", "latin")))
    return results

def compute_mrr(ranks, k=10):
    vals = [(1/r if r is not None and r <= k else 0.0) for r in ranks]
    return float(np.mean(vals)) if vals else 0.0

def compute_recall(ranks, k):
    vals = [(1 if r is not None and r <= k else 0) for r in ranks]
    return float(np.mean(vals)) if vals else 0.0

def compute_ndcg(ranks, k):
    vals = []
    for r in ranks:
        if r is not None and r <= k:
            vals.append(1 / np.log2(r + 1))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0

def aggregate(ranks, ks):
    agg = {
        "MRR": compute_mrr(ranks, k=10),
        **{f"Recall@{k}": compute_recall(ranks, k) for k in ks},
        "NDCG@10": compute_ndcg(ranks, 10),
    }
    return agg

def load_retriever(name, checkpoint=None, device="cuda"):
    if name == "model":
        from src.eval.model_retriever import build_index as _build_index, retrieve
        def build_index(corpus):
            return _build_index(corpus, checkpoint, device)
        return build_index, retrieve
    elif name == "levenshtein":
        from src.eval.baselines.levenshtein import build_index, retrieve
    elif name == "soundex":
        from src.eval.baselines.soundex import build_index, retrieve
    elif name == "bm25":
        from src.eval.baselines.bm25 import build_index, retrieve
    elif name == "transliterate":
        from src.eval.baselines.transliterate import build_index, retrieve
    else:
        raise ValueError(f"Unknown retriever: {name}")
    return build_index, retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Parallel workers for CPU retrievers. Ignored for model retriever.")
    args = parser.parse_args()

    ks = sorted(args.k)
    max_k = max(ks)
    records = []

    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                records.append(r)

    corpus = [
        {
            "entity_id": r["entity_id"],
            "text": r["anchor"]["text"],
            "bytes": r["anchor"].get("bytes")
        }
        for r in records
    ]

    build_index, retrieve = load_retriever(args.retriever, args.checkpoint, args.device)
    print("Building index...")
    index = build_index(corpus)
    print("Index built.")

    queries = []
    for r in records:
        for p in r["positives"]:
            queries.append({
                "entity_id": r["entity_id"],
                "text": p["text"],
                "bytes": p.get("bytes"),
                "type": p["type"],
                "script": p.get("script", "latin"),
            })

    results = {
        "overall": [],
        "by_type": {
            "phonetic": [],
            "script": [],
            "combined": [],
        },
        "by_script": {},
    }

    print(f"Running {len(queries)} queries...")
    if args.retriever in PARALLEL_RETRIEVERS:
        from pqdm.processes import pqdm as pqdm_proc
        n_jobs = args.n_jobs or os.cpu_count()
        chunk_size = max(1, len(queries) // n_jobs)
        chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]
        job_args = [(chunk, index, max_k, retrieve) for chunk in chunks]
        chunk_results = pqdm_proc(job_args, _run_query_batch, n_jobs=n_jobs)
        query_results = []
        for chunk in chunk_results:
            if isinstance(chunk, Exception):
                raise chunk
            query_results.extend(chunk)
    else:
        query_results = []
        for q in tqdm(queries):
            retrieved = retrieve(q["text"], q["bytes"], index, max_k)
            rank = None
            for i, eid in enumerate(retrieved):
                if eid == q["entity_id"]:
                    rank = i + 1
                    break
            query_results.append((rank, q["type"], q.get("script", "latin")))

    for result in query_results:
        if isinstance(result, Exception):
            raise result
        rank, qtype, script = result
        results["overall"].append(rank)
        if qtype in results["by_type"]:
            results["by_type"][qtype].append(rank)
        if script not in results["by_script"]:
            results["by_script"][script] = []
        results["by_script"][script].append(rank)

    output = {
            "retriever": args.retriever,
            "checkpoint": args.checkpoint,
            "k": ks,
            "n_queries": len(queries),
            "n_entities": len(corpus),
            "overall": aggregate(results["overall"], ks),
            "by_type": {
                k: aggregate(v, ks) for k,v in results["by_type"].items()
            },
            "by_script": {
                k: aggregate(v, ks) for k,v in results["by_script"].items()
            }
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()