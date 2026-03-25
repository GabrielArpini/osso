import glob

import torch
from osso.config import ModelConfig, RoPEConfig
from osso.models.llama import LlamaModel, remap_weights
from safetensors.torch import load_file
from osso.utils.hf import cached_load_hf_config, download_hf_weight, load_tokenizer


def load_model_config(model_path: str) -> ModelConfig:
    hf_cfg = cached_load_hf_config(model_path)
    return ModelConfig(
        n_layers=hf_cfg.num_hidden_layers,
        hidden_size=hf_cfg.hidden_size,
        head_dim=hf_cfg.head_dim,
        num_qo_heads=hf_cfg.num_attention_heads,
        num_kv_heads=hf_cfg.num_key_value_heads,
        ffn_dim=hf_cfg.intermediate_size,
        vocab_size=hf_cfg.vocab_size,
        rms_eps=hf_cfg.rms_norm_eps,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        rotary_config=RoPEConfig(
            dim=hf_cfg.head_dim,
            end=hf_cfg.max_position_embeddings,
            theta=getattr(hf_cfg, "rope_theta", 500000.0),
        ),
    )


def load_weights(weights_dir: str) -> dict:
    state_dict = {}
    for path in sorted(glob.glob(f"{weights_dir}/*.safetensors")):
        state_dict.update(load_file(path))
    return state_dict


class Engine:
    def __init__(self, model_path: str):
        self.tokenizer = load_tokenizer(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        config = load_model_config(model_path)
        weights_dir = download_hf_weight(model_path)
        state_dict = load_weights(weights_dir)
        state_dict = remap_weights(state_dict, config)

        self.model = LlamaModel(config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, self.dtype)
        self.model.eval()
