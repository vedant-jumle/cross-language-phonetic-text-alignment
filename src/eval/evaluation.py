import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path

def computer_mrr(ranks, k=10):
    vals = [(1/r if r is not None and r <= k else 0.0) for r in ranks]
    return float(np.mean(vals)) if vals else 0.0

def computer_recall(ranks, k):
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
        "MRR": computer_mrr(ranks, k=10),
        **{f"Recall@{k}": computer_recall(ranks, k) for k in ks},
        "NDC@10": compute_ndcg(ranks, 10),
    }
    return agg

def load_retriever(name):
    if name == "model":
        from src.eval.model_retriever import build_index, retrieve
    elif name == "levenshtein":
        from src.eval.baselines.levenshtein import build_index, retrieve
    elif name == "soundex":
        from src.eval.baselines.soundex import build_index, retrieve
    elif name == "bm25":
        from src.eval.baselines.bm25 import build_index, retrieve
    elif name == "transliterate":
        from src.eval.baselines.transliterate import build_index, retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
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

    build_index, retrieve = load_retriever(args.retriever)
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
            "script+phonetic": [],
        },
        "by_script": {},
    }

    print(f"Running {len(queries)} queries...")
    for q in tqdm(queries):
        retrieved = retrieve(q["text"], q["bytes"], index, max_k)
        rank = None
        for i, eid in enumerate(retrieved):
            if eid == q["entity_id"]:
                rank = i+1
                break
        results["overall"].append(rank)

        if q["type"] in results["by_type"]:
            results["by_type"][q["type"]].append(rank)
            
        script = q.get("script") or "latin"
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
                k: aggregate(v, ks) for k,v in results["by_type"].items
            },
            "by_script": {
                k: aggregate(v, ks) for k,v in results["by_script"].items
            }
        }

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()