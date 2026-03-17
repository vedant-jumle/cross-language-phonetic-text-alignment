from __future__ import annotations

import numpy as np
import torch

from src.llm_generator_pipeline.dataloader import pad_sequences
from src.llm_generator_pipeline.dataset import NameDataset

from .encoder import ByteLevelEncoder


def encode_all(
    model: ByteLevelEncoder,
    dataset: NameDataset,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode all anchor sequences in dataset.

    Returns:
        embeddings: float32 numpy array of shape (N, hidden_dim), L2-normalized
        entity_ids: list of str, same order as embeddings
    """
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_entity_ids: list[str] = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_records = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            seqs = [r["anchor"]["bytes"] for r in batch_records]
            input_ids, attention_mask = pad_sequences(seqs, pad_value=0)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            embeddings = model(input_ids, attention_mask)  # (B, D)
            all_embeddings.append(embeddings.cpu().numpy())
            all_entity_ids.extend(r["entity_id"] for r in batch_records)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32), all_entity_ids
