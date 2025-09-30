import math

import torch
import torch.nn as nn
from einops import einsum, reduce
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
        device=None,
        dtype: None = None,
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

        rms = torch.rsqrt(reduce(x.pow(2), "... d_model -> ... 1", "mean") + self.eps)
        result = self.weight * (x * rms)

        return result.to(in_dtype)
