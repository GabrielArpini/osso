from __future__ import annotations

import torch
from osso.config import ModelConfig
from osso.kvcache.base import BaseKVCache


class NaiveKVCache(BaseKVCache):
    def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype) -> None:
        self._kv_buffer = torch.zeros(
            (2, config.n_layers, config.batch_size, config.max_seq_len, config.num_kv_heads, config.head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._seq_len = 0

    def k_cache(self, layer_id: int) -> torch.Tensor:
        return self._k_buffer[layer_id][:, :self._seq_len]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self._v_buffer[layer_id][:, :self._seq_len]

    def store(self, layer_id: int, start_pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        new_len = k.shape[1]
        self._k_buffer[layer_id][:, start_pos:start_pos + new_len] = k
        self._v_buffer[layer_id][:, start_pos:start_pos + new_len] = v
        self._seq_len = start_pos + new_len

    @property
    def seq_len(self) -> int:
        return self._seq_len
