from metaphone import doublemetaphone
from typing import Any
import editdistance


def build_index(corpus: list[dict]) -> Any:
    codes = {}

    texts = {entry["entity_id"]: entry["text"] for entry in corpus}

    for entry in corpus:
        for code in doublemetaphone(entry["text"]):
            if not code:
                continue
            if code not in codes:
                codes[code] = []
            codes[code].append(entry["entity_id"])

    return {"texts": texts, "codes": codes}



def retrieve(query_text: str, query_bytes: list[int], index: Any, k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    
    # has code 1 and code 2
    codes = doublemetaphone(query_text)
    candidates = set()
    results = []
    query_codes = []
    for code in codes:
        if code:
            query_codes.append(code)

    for code in query_codes:
        if code in index["codes"]:
            candidates.update(index["codes"][code])

    for entity_id in candidates:
        text = index["texts"][entity_id]
        dist = editdistance.eval(query_text, text)
        results.append((dist, entity_id))

    results.sort(key=lambda x: x[0])

    ranked_results = [entity_id for _, entity_id in results]

    if len(ranked_results) < k:
        fallback_results = []
        for entity_id, text in index["texts"].items():
            if entity_id in candidates:
                continue
            dist = editdistance.eval(query_text, text)
            fallback_results.append((dist, entity_id))
        fallback_results.sort(key=lambda x: x[0])
        ranked_results.extend([entity_id for _, entity_id in fallback_results])
        
    return ranked_results[:k]
