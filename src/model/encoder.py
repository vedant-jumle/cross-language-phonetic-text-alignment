import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import ByteEncoderConfig


class ByteLevelEncoder(PreTrainedModel):
    config_class = ByteEncoderConfig

    def __init__(self, config: ByteEncoderConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
            padding_idx=config.pad_token_id,
        )
        self.pos_embedding = nn.Embedding(config.max_len, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable from scratch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.post_init()  # HF weight init

    def forward(
        self,
        input_ids: torch.LongTensor,       # (B, L)
        attention_mask: torch.BoolTensor,  # (B, L), True = real token
    ) -> torch.FloatTensor:               # (B, D), L2-normalized

        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)

        x = self.embedding(input_ids) + self.pos_embedding(positions)  # (B, L, D)

        # TransformerEncoder: src_key_padding_mask True = IGNORE → invert mask
        padding_mask = ~attention_mask  # (B, L)

        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, L, D)

        # mean pool over real tokens only
        mask_f = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (x * mask_f).sum(dim=1)               # (B, D)
        counts = mask_f.sum(dim=1).clamp(min=1)        # (B, 1)
        pooled = summed / counts                        # (B, D)

        return F.normalize(pooled, p=2, dim=-1)         # (B, D), unit vectors
