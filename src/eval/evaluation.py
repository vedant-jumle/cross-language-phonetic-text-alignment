import numpy as np

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