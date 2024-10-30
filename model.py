import torch


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        torch.nn.Transformer

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class GPT2(torch.nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()

        self.word_embedding = torch.nn.Embedding(50257, 768)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(768, 12, batch_first=True),
            num_layers=12,
        )
        self.positional_embedding = SinusoidalPositionalEmbedding(768)
        self.classifier = torch.nn.Linear(768, 50257)

    def forward(
        self,
        x,
        key_padding_mask: torch.Tensor,
    ):
        assert x.size(1) <= 1024
        x = self.word_embedding(x)
        x = self.positional_embedding(x)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        x = self.transformer_encoder(
            x, src_key_padding_mask=key_padding_mask.float(), is_causal=True, mask=mask
        )
        x = self.classifier(x)
        return x
