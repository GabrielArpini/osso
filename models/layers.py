import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, eps, size):
        super().__init__()
        self.eps = eps
        self.size = size
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        rms = torch.rsqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

# Re implementation of original llama rope, but with einops from: https://github.com/meta-llama/llama/pull/1173
def RoPE(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(rearrange(xq.float(), '... (c d) -> ... c d', d=2))
    xk_ = torch.view_as_complex(rearrange(xk.float(), '... (c d) -> ... c d', d=2))
    freqs_cis = rearrange(freqs_cis, 's d -> 1 s 1 d')
    xq_out = rearrange(torch.view_as_real(xq_ * freqs_cis), '... c d -> ... (c d)')
    xk_out = rearrange(torch.view_as_real(xk_ * freqs_cis), '... c d -> ... (c d)')
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
