import torch
from config import ModelConfig
from layers.attention import Attention
from layers.utils import RMSNorm, SwiGLU
from torch import nn


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = Attention(config)
        self.norm = RMSNorm(config.rms_norm_eps, config.hidden_size)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.feed_forward = SwiGLU(config.hidden_size, config.ffn_dim)
        self.ffn_norm = RMSNorm(config.rms_norm_eps, config.hidden_size)

    def forward(self, x):
        h = self.attention(x)
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

    def forward(self, tokens):
        x = self.tok_embeddings(tokens)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def remap_weights(state_dict: dict, config: ModelConfig) -> dict:
    new_sd = {}
    new_sd["tok_embeddings.weight"] = state_dict["model.embed_tokens.weight"]
    new_sd["norm.weight"] = state_dict["model.norm.weight"]
    new_sd["output.weight"] = state_dict["lm_head.weight"]

    for i in range(config.n_layers):
        hf = f"model.layers.{i}"
        m = f"blocks.{i}"

        # fuse q, k, v projections into wqkv
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
