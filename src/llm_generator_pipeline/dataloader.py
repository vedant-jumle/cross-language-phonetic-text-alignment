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

def collate_fn(batch: list[tuple[list[int], list[int], list[int]]]) -> dict[str, torch.Tensor]:
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
    
class HardNegativeSampler:
    def __init__(self, dataset: NameDataset, config: dict[str, Any]):
        self.dataset = dataset
        self.config = config
        
        self.warmup = int(config["hard_negative_warmup_steps"])
        self.mix_ratio = float(config["hard_negative_mix_ratio"])
        
        if not (0.0 <= self.mix_ratio <= 1.0):
            raise ValueError(f"hard_negative_mix_ratio must be [0,1], got {self.mix_ratio}")
        
        self._step = 0
        self.index: faiss.Index | None = None
        self.index_entity_ids: list[str] = []
        self.entity_id_to_index_pos: dict[str, int] = {}
        self.entity_id_to_record: dict[str, dict[str, Any]] = {record["entity_id"]: record for record in self.dataset.records}
        self.rng = random.Random()
        
    def step(self) -> None:
        self._step += 1
        
    def update_index(self, embeddings: np.ndarray, entity_ids: list[str]) -> None:
        if len(entity_ids) != len(embeddings):
            raise ValueError(
                f"Length mismatch: len(entity_ids)={len(entity_ids)} vs "
                f"len(embeddings)={len(embeddings)}"
            )
            
        if len(entity_ids) == 0:
            self.index = None
            self.index_entity_ids = []
            self.entity_id_to_index_pos = {}
            return
        
        
        filtered_embeddings = []
        filtered_entity_ids = []
        
        for embedding, entity_id in zip(embeddings, entity_ids):
            if entity_id in self.entity_id_to_record:
                filtered_embeddings.append(embedding)
                filtered_entity_ids.append(entity_id)
                
        if not filtered_entity_ids:
            self.index = None
            self,self.index_entity_ids = []
            self.entity_id_to_index_pos = {}
            return
        
        emb_array = np.asarray(filtered_embeddings, dtype=np.float32)
        if emb_array.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array of shape (N, D), got shape={emb_array.shape}")
        
        faiss.normalize_L2(emb_array)
        
        dim = emb_array.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_array)
        
        self.index = index
        self.index_entity_ids = filtered_entity_ids
        self.entity_id_to_index_pos = {entity_id: i for i, entity_id in enumerate(filtered_entity_ids)}
        
    def sample_negatives(self, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        negatives: list[dict[str, Any]] = []
        
        constraint = (self._step >= self.warmup and self.index is not None and len(self.index_entity_ids) > 1)
        for anchor in anchors:
            use_hard = constraint and (self.rng.random() < self.mix_ratio)
            
            if use_hard:
                neg = self.sample_hard_negatives(anchor)
                if neg is None:
                    neg = self.sample_random_negative(anchor["entity_id"])
            else:
                neg = self.sample_random_negative(anchor["entity_id"])
                
            negatives.append(neg)
            
        return negatives
    
    def sample_random_negative(self, anchor_entity_id: str) -> dict[str, Any]:
        if len(self.dataset.records) < 2:
            raise ValueError("Dataset must contain at least 2 records.")
        
        while True:
            candidate = self.rng.choice(self.dataset.records)
            if candidate["entity_id"] != anchor_entity_id:
                return candidate
        
    def sample_hard_negatives(self, anchor: dict[str, Any], top_k: int = 10) -> dict[str, Any] | None:
        if self.index is None:
            return None
        
        anchor_entity_id = anchor["entity_id"]
        anchor_pos = self.entity_id_to_index_pos.get(anchor_entity_id)
        
        if anchor_pos is None:
            return None
        
        query = np.zeros((1, self.index.d), dtype=np.float32)
        self.index.reconstruct(anchor_pos, query[0])
        
        k = min(top_k + 1, len(self.index_entity_ids))
        _, indices = self.index.search(query, k)
        
        candidate_records: list[dict[str, Any]] = []
        for index in indices[0]:
            if index < 0:
                continue
            candidate_entity_id = self.index_entity_ids[index]
            if candidate_entity_id == anchor_entity_id:
                continue
            
            candidate = self.entity_id_to_record.get(candidate_entity_id)
            if candidate is None:
                continue
            
            candidate_records.append(candidate)
            
        if not candidate_records:
            return None
        
        return self.rng.choice(candidate_records)
    
def choose_positive_bytes(record: dict[str, Any], rng: random.Random) -> list[int]:
    positives = record.get("positives", [])
    
    if len(positives) == 0:
        return record["anchor"]["bytes"]   

    chosen = rng.choice(positives)
    return chosen["bytes"]

def build_dataloader(dataset: NameDataset, sampler: HardNegativeSampler, config: dict[str, Any], shuffle: bool = True) -> DataLoader:
    batch_size = int(config["batch_size"])
    rng = random.Random()
    
    def triplet_collate(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        negatives = sampler.sample_negatives(records)
        
        triplets: list[tuple[list[int], list[int], list[int]]] = []
        for anchor_record, negative_record in zip(records, negatives):
            anchor_bytes = anchor_record["anchor"]["bytes"]
            positive_bytes = choose_positive_bytes(anchor_record, rng)
            negative_bytes = negative_record["anchor"]["bytes"]
            
            triplets.append((anchor_bytes, positive_bytes, negative_bytes))
        return collate_fn(triplets)
            
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=triplet_collate)
        