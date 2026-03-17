from functools import partial

import torch
import torch.nn.functional as F


def infonce_loss(
    anchor: torch.FloatTensor,              # (B, D)
    positive: torch.FloatTensor,            # (B, D)
    negative: torch.FloatTensor = None,     # accepted but ignored — in-batch negatives used instead
    temperature: float = 0.07,
) -> torch.FloatTensor:
    B = anchor.shape[0]
    logits = (anchor @ positive.T) / temperature   # (B, B)
    labels = torch.arange(B, device=anchor.device)  # diagonal = correct pair
    return F.cross_entropy(logits, labels)


def triplet_loss(
    anchor: torch.FloatTensor,    # (B, D)
    positive: torch.FloatTensor,  # (B, D)
    negative: torch.FloatTensor,  # (B, D)
    margin: float = 0.3,
) -> torch.FloatTensor:
    d_pos = 1 - (anchor * positive).sum(dim=-1)   # cosine distance
    d_neg = 1 - (anchor * negative).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


def build_loss(config: dict):
    loss_type = config["loss"]
    if loss_type == "infonce":
        return partial(infonce_loss, temperature=float(config["temperature"]))
    elif loss_type == "triplet":
        return partial(triplet_loss, margin=float(config["margin"]))
    raise ValueError(f"Unknown loss type: {loss_type!r}. Must be 'infonce' or 'triplet'.")
