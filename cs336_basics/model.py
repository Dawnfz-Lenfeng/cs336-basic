import math

import einx
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .nn_utils import softmax


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
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((vocab_size, d_model), device=device, dtype=dtype),
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
        output = self._norm(x.float()).type_as(x)

        return self.weight * output

    def _norm(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        return x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)


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
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        cos, sin = self._precompute_freqs(theta, d_k, max_seq_len, device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(
        x: Float[Tensor, " ... d_k"],
    ) -> Float[Tensor, " ... d_k"]:
        x1, x2 = rearrange(x, "... (d_k_half pair) -> pair ... d_k_half", pair=2)
        return einx.rearrange(
            "... d_k_half, ... d_k_half -> ... (d_k_half 1 + 1)", -x2, x1
        )

    @staticmethod
    def _precompute_freqs(
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[Float[Tensor, "max_seq_len d_k"], Float[Tensor, "max_seq_len d_k"]]:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        freqs = torch.outer(positions, inv_freq)
        freqs = repeat(freqs, "seq d_k_half -> seq (d_k_half pair)", pair=2)

        return freqs.cos(), freqs.sin()


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    scores = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(~mask, -1e9)

    attention_weights = softmax(scores, dim=-1)
    output = einsum(
        attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )

    return output


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.size(-2)
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)

        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        output = scaled_dot_product_attention(Q, K, V, mask)
        output = rearrange(output, "... h seq_len d_k -> ... seq_len (h d_k)")

        return self.output_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len)

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGlu(d_model, d_ff)

    def forward(
        self, x: Float[Tensor, " batch seq_len d_model"]
    ) -> Float[Tensor, " batch seq_len d_model"]:
        seq_len = x.size(-2)
        token_positions = torch.arange(seq_len, device=x.device)

        x += self.attn(self.ln1(x), token_positions)
        x += self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        )

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        x: Int[Tensor, " batch seq_len"],
    ) -> Float[Tensor, " batch seq_len vocab"]:
        x = self.token_embeddings(x)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
