from dataclasses import asdict

import torch
from osso.config import ModelConfig
from osso.kvcache.base import BaseKVCache
from osso.layers.attention import Attention
from osso.layers.utils import RMSNorm, SwiGLU, precompute_cos_sin
from torch import nn


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = Attention(config)
        self.norm = RMSNorm(config.rms_norm_eps, config.hidden_size)

    def forward(self, x, layer_id: int = 0, kv_cache: BaseKVCache | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None):
        return x + self.attn(self.norm(x), layer_id, kv_cache, position_embeddings)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.feed_forward = SwiGLU(config.hidden_size, config.ffn_dim)
        self.ffn_norm = RMSNorm(config.rms_norm_eps, config.hidden_size)

    def forward(self, x, layer_id: int = 0, kv_cache: BaseKVCache | None = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None):
        h = self.attention(x, layer_id, kv_cache, position_embeddings)
        return h + self.feed_forward(self.ffn_norm(h))


class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.rms_norm_eps, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        cos, sin = precompute_cos_sin(**asdict(config.rotary_config))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens, kv_cache: BaseKVCache | None = None):
        x = self.tok_embeddings(tokens)
        start_pos = kv_cache.seq_len if kv_cache is not None else 0
        seqlen = x.shape[1]
        position_embeddings = (
            self.cos[start_pos:start_pos + seqlen],
            self.sin[start_pos:start_pos + seqlen],
        )
        for layer_id, block in enumerate(self.blocks):
            x = block(x, layer_id, kv_cache, position_embeddings)
        return self.output(self.norm(x))


def remap_weights(state_dict: dict, config: ModelConfig) -> dict:
    new_sd = {}
    new_sd["tok_embeddings.weight"] = state_dict["model.embed_tokens.weight"]
    new_sd["norm.weight"] = state_dict["model.norm.weight"]
    if "lm_head.weight" in state_dict:
        new_sd["output.weight"] = state_dict["lm_head.weight"]
    else:
        new_sd["output.weight"] = state_dict["model.embed_tokens.weight"]

    for i in range(config.n_layers):
        hf = f"model.layers.{i}"
        m = f"blocks.{i}"

        q = state_dict[f"{hf}.self_attn.q_proj.weight"]
        k = state_dict[f"{hf}.self_attn.k_proj.weight"]
        v = state_dict[f"{hf}.self_attn.v_proj.weight"]
        new_sd[f"{m}.attention.attn.wqkv.weight"] = torch.cat([q, k, v], dim=0)

        new_sd[f"{m}.attention.attn.wo.weight"] = state_dict[f"{hf}.self_attn.o_proj.weight"]
        new_sd[f"{m}.attention.norm.weight"] = state_dict[f"{hf}.input_layernorm.weight"]

        new_sd[f"{m}.feed_forward.gate_proj.weight"] = state_dict[f"{hf}.mlp.gate_proj.weight"]
        new_sd[f"{m}.feed_forward.up_proj.weight"] = state_dict[f"{hf}.mlp.up_proj.weight"]
        new_sd[f"{m}.feed_forward.down_proj.weight"] = state_dict[f"{hf}.mlp.down_proj.weight"]
        new_sd[f"{m}.ffn_norm.weight"] = state_dict[f"{hf}.post_attention_layernorm.weight"]

    return new_sd
