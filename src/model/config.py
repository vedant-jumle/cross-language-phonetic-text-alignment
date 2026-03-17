from transformers import PretrainedConfig


class ByteEncoderConfig(PretrainedConfig):
    model_type = "byte_encoder"

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.max_len = max_len
