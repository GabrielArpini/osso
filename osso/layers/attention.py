from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from einops import rearrange
from utils import RMSNorm, apply_rope, precompute_freqs_cis


# Repeat kv from official llama repo.
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(
        self, config: ModelConfig, q_norm: RMSNorm | None = None, k_norm: RMSNorm | None = None
    ):
        super().__init__()
        assert config.num_qo_heads % config.num_kv_heads == 0
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_qo_heads // self.num_kv_heads
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.wqkv = nn.Linear(config.hidden_size, (self.num_qo_heads + 2 * self.num_kv_heads) * config.head_dim, bias=False)
        self.wo = nn.Linear(self.num_qo_heads * config.head_dim, config.hidden_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(**asdict(config.rotary_config))

    def forward(self, x):
        _, seqlen, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(
            [self.num_qo_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim],
            dim=-1,
        )
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_qo_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_kv_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_kv_heads)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        q, k = apply_rope(q, k, self.freqs_cis[:seqlen])
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = rearrange(o, "b h s d -> b s (h d)")
        return self.wo(o)
