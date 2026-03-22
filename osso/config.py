from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoPEConfig:
    dim: int
    end: int
    theta: float


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    head_dim: int
    vocab_size: int
    rms_eps: float
    hidden_size: int
    rms_norm_eps: float
    num_qo_heads: int
    num_kv_heads: int
    ffn_dim: int
    rotary_config: RoPEConfig
