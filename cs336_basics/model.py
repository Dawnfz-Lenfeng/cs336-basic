import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat
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

        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    (out_features, in_features),
                    device=device,
                    dtype=dtype,
                ),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

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

        std = 1.0
        self.weight: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(
                    (num_embeddings, embedding_dim), device=device, dtype=dtype
                ),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

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

        self.eps = eps
        self.weight: Float[Tensor, " d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        inv_rms: Float[Tensor, " ... 1"] = (
            reduce(x.pow(2), "... d_model -> ... 1", "mean").add(self.eps).rsqrt()
        )

        return (self.weight * x * inv_rms).to(in_dtype)


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
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.register_buffer(
            "cos_sin",
            self._get_cos_sin(theta, d_k, max_seq_len).to(device),
            persistent=False,
        )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos, sin = self.cos_sin[:, token_positions]
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)

        return x_rotated

    @staticmethod
    def _rotate_half(
        x: Float[Tensor, " ... d_k"],
    ) -> Float[Tensor, " ... d_k"]:
        x1, x2 = rearrange(x, "... (d_k_half pair) -> pair ... d_k_half", pair=2)
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d_k_half pair -> ... (d_k_half pair)"
        )

    @staticmethod
    def _get_cos_sin(
        theta: float,
        d_k: int,
        max_seq_len: int,
    ) -> Float[Tensor, " 2 max_seq_len d_k"]:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len)

        freqs = einsum(
            positions,
            inv_freq,
            "max_seq_len, d_k_half -> max_seq_len d_k_half",
        )
        freqs = repeat(freqs, "max_seq_len d_k_half -> max_seq_len (d_k_half 2)")

        return torch.stack((freqs.cos(), freqs.sin()))
