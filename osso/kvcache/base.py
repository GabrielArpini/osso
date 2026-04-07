from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseKVCache(ABC):
    """Abstract KV Cache, to make it easier to create multiple variations"""

    @abstractmethod
    def k_cache(self, layer_id: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, layer_id: int) -> torch.Tensor: ...

    @abstractmethod
    def store(self, layer_id: int, start_pos: int, k: torch.Tensor, v: torch.Tensor) -> None: ...

    @property
    @abstractmethod
    def seq_len(self) -> int: ...
