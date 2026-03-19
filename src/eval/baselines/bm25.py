from rank_bm25 import BM25Okapi
from typing import Any

def char_ngrams(text: str, n: int = 3) -> list[str]:
    # padding with spaces to capture word boundaries
    text = f"  {text.lower()}  "  
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def build_index(corpus: list[dict]) -> Any:
    # Tokenize each corpus text into character trigrams
    # Fit BM25 over the tokenized corpus
    tokenized = []
    entity_ids = []

    for entry in corpus:
        tokenized.append(char_ngrams(entry["text"], n = 3))
        entity_ids.append(entry["entity_id"])

    bm25 = BM25Okapi(tokenized)

    return {"bm25": bm25, "entity_ids": entity_ids}



def retrieve(query_text: str, query_bytes: list[int], index: Any, k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    results = []

    query_tokens = char_ngrams(query_text, n = 3)

    scores = index["bm25"].get_scores(query_tokens)
    results = sorted(zip(scores, index["entity_ids"]), key=lambda x: x[0], reverse=True)
    return [entity_id for _, entity_id in results[:k]]