# Stack

## Runtime

- **Language:** Python 3.12
- **GPU:** NVIDIA GPU stack (CUDA)

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.10.0 | Deep learning framework (PyTorch) |
| `transformers` | 5.3.0 | Hugging Face model loading and inference |
| `nvidia-ml-py` | 13.590.48 | VRAM monitoring via NVIDIA NVML |
| `wandb` | 0.25.0 | Experiment tracking (Weights & Biases) |
| `openai` | 2.26.0 | OpenAI-compatible API client (vLLM server testing) |
| `einops` | — | Tensor operations |
| `accelerate` | — | HF training acceleration |
| `huggingface-hub` | — | Model downloading from HF Hub |
| `httpx` | — | Async HTTP client (health check polling) |
| `weave` | — | W&B tracing instrumentation |

## Configuration

- Dependencies managed via `uv` (see `pyproject.toml`)
- GPU required for model inference
- Models loaded from Hugging Face Hub

## Build / Package

- `pyproject.toml` — project metadata and dependencies
- `uv.lock` — locked dependency versions

---
*Generated: 2026-03-11*
