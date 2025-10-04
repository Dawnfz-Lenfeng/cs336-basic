import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight: Float[Tensor, " d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms: Float[Tensor, " ... 1"] = (
            reduce(x.pow(2), "... d_model -> ... 1", "mean").add(self.eps).rsqrt()
        )
        x_normalized: Float[Tensor, " ... d_model"] = x * rms
        result: Float[Tensor, " ... d_model"] = self.weight * x_normalized

        return result.to(in_dtype)


class SwiGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        gate: Float[Tensor, " ... d_ff"] = F.silu(self.w1(x))
        filtered: Float[Tensor, " ... d_ff"] = gate * self.w3(x)

        return self.w2(filtered)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq.to(device))

        cos, sin = self._get_cos_sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_rotated = (x * cos) + (self.rotate_half(x) * sin)

        return x_rotated

    @staticmethod
    def rotate_half(
        x: Float[Tensor, " ... d_k"],
    ) -> Float[Tensor, " ... d_k"]:
        return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)

    def _get_cos_sin(
        self
    ) -> tuple[Float[Tensor, " max_seq_len d_k"], Float[Tensor, " max_seq_len d_k"]]:
        positions = torch.arange(self.max_seq_len, device=self.device)
        freqs = einsum(
            positions,
            self.inv_freq,
            "max_seq_len, d_k_half -> max_seq_len d_k_half",
        )
        freqs = repeat(freqs, "max_seq_len d_k_half -> max_seq_len (d_k_half 2)")

        return freqs.cos(), freqs.sin()
