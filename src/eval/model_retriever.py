import numpy as np
import faiss
import torch
from tqdm import tqdm

from src.model.encoder import ByteLevelEncoder

encode_batch_size = 512
index_type = "flat"

def resolve_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    
    return "cpu"

def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def encode_batch(model: ByteLevelEncoder, batch_bytes: list[list[int]], device: str) -> np.ndarray:
    if not batch_bytes:
        return np.empty((0,0), dtype=np.float32)
    
    max_len = max(len(x) for x in batch_bytes)
    batch_size = len(batch_bytes)
    
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, seq in enumerate(batch_bytes):
        seq_len = len(seq)
        if seq_len == 0:
            continue
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = True
        
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, torch.Tensor):
            embeddings = outputs
        elif isinstance(outputs, (tuple, list)):
            embeddings = outputs[0]
        elif hasattr(outputs, "embeddings"):
            embeddings = outputs.embeddings
        elif hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError("Unsupported model output type!")
        
        embeddings = l2_norm(embeddings)
        embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
        
    return embeddings

def encode_single_query(model:ByteLevelEncoder, query_bytes: list[int], device: str) -> np.ndarray:
    max_len = getattr(model.config, "max_len", None)
    if max_len is not None and len(query_bytes) > max_len:
        query_bytes = query_bytes[:max_len]
        
    input_ids = torch.tensor(query_bytes, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, torch.Tensor):
            embedding = outputs
        elif isinstance(outputs, (tuple, list)):
            embedding = outputs[0]
        elif hasattr(outputs, "embeddings"):
            embedding = outputs.embeddings
        elif hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError("Unsupported model output type!")
        
        embedding = l2_norm(embedding)
        embedding = embedding.detach().cpu().numpy().astype(np.float32)
    
    return embedding

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings. Instead, got shape {embeddings.shape}")
    
    dim = embeddings.shape[1]
    
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "ivf_flat":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    elif index_type == "ivf_pq":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, 100, 8, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    else: 
        raise ValueError(f"Unsupported index_type: {index_type}")
    
    index.add(embeddings)
    
    return index

def build_index(corpus: list[dict], checkpoint_dir: str, device: str):
    resolved_device = resolve_device(device)
    
    model = ByteLevelEncoder.from_pretrained(checkpoint_dir)
    model.eval()
    model.to(resolved_device)
    
    entity_ids = [item["entity_id"] for item in corpus]
    
    all_embeddings = []
    for start in tqdm(range(0, len(corpus), encode_batch_size), desc="Encoding corpus..."):
        batch = corpus[start:start + encode_batch_size]
        batch_bytes = [item["bytes"] for item in batch]
        batch_embeddings = encode_batch(model, batch_bytes, resolved_device)
        all_embeddings.append(batch_embeddings)
        
    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    else:
        raise ValueError("Corpus is empty? Cannot build retrieval empty without anything!")
    
    faiss_index = build_faiss_index(embeddings)
    
    return {
        "faiss_index": faiss_index,
        "entity_ids": entity_ids,
        "model": model,
        "device": resolved_device,
    }
    
    
def retrieve(query_text: str, query_bytes: list[int], index, k: int) -> list[str]:
    del query_text
    
    model = index["model"]
    device = index["device"]
    faiss_index = index["faiss_index"]
    entity_ids = index["entity_ids"]
    
    query_embedding = encode_single_query(model, query_bytes, device)
    _, indices = faiss_index.search(query_embedding, k)
    
    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(entity_ids[idx])
        
    return results

