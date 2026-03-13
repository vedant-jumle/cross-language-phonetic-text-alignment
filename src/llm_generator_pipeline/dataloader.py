from __future__ import annotations

import random
from typing import Any

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import NameDataset

def pad_sequences(sequences: list[list[int]], pad_value: int = 0) -> tuple[torch.LongTensor, torch.BoolTensor]:
    if not sequences:
        raise ValueError("Cannot pad an empty list.")
    
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        padded[i, : len(seq)] = seq_tensor
        mask[i, : len(seq)] = True
        
    return padded, mask

def collate(batch: list[tuple[list[int], list[int], list[int]]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Received empty batch.")
    
    anchor_seqs = [item[0] for item in batch]
    positive_seqs = [item[1] for item in batch]
    negative_seqs = [item[2] for item in batch]
    
    anchor, anchor_mask = pad_sequences(anchor_seqs, pad_value=0)
    positive, positive_mask = pad_sequences(positive_seqs, pad_value=0)
    negative, negative_mask = pad_sequences(negative_seqs, pad_value=0)
    
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "anchor_mask": anchor_mask,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
    }
    
