from src.eval.model_retriever import build_index, retrieve


corpus = [
    {"entity_id": "a", "text": "doc a", "bytes": [1, 2, 3, 4]},
    {"entity_id": "b", "text": "doc b", "bytes": [10, 11, 12, 13]},
    {"entity_id": "c", "text": "doc c", "bytes": [1, 2, 3, 5]},
]

index = build_index(
    corpus=corpus,
    checkpoint_dir="checkpoints/best",
    device="cuda",
)

results = retrieve(
    query_text="ignored",
    query_bytes=[1, 2, 3, 4],
    index=index,
    k=2,
)

print(results)

# from the project root, run python -m src.eval.test