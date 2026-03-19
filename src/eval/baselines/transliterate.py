import editdistance
from typing import Any
from icu import Transliterator

t = Transliterator.createInstance("Any-Latin; Latin-ASCII")

def build_index(corpus: list[dict]) -> Any:
    # list of entity_id and text pairs
    # no preprocessing
    return [(entry['entity_id'], entry['text']) for entry in corpus]


def retrieve(query_text: str, query_bytes: list[int], index: Any, k: int) -> list[str]:
    # computing edit distance from query_text to every corpus text
    # returning top-k entity_ids sorted by ascending distance
    if k <= 0:
        raise ValueError("k must be a positive integer")
    results = []
    latin_query = t.transliterate(query_text)
    for entity_id, text in index:
        # levenshthein distance
        dist = editdistance.eval(latin_query, text)
        results.append((dist, entity_id))
    results.sort(key=lambda x: x[0])
    return [entity_id for _, entity_id in results[:k]]




