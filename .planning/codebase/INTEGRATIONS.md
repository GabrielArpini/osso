# Integrations

## External Services

### OpenAI-Compatible API (vLLM)
- **Purpose:** Server-based inference testing
- **Used in:** `tests/benchmark.py`
- **Details:** OpenAI SDK (v2.26.0) pointed at a local vLLM server endpoint

### Weights & Biases (wandb)
- **Purpose:** Experiment tracking, metric logging
- **Used in:** Training/evaluation scripts
- **Details:** Project initialization and metric logging via `wandb` SDK (v0.25.0)

### Weave (W&B)
- **Purpose:** Tracing and instrumentation
- **Details:** W&B's Weave library for trace capture

### Hugging Face Model Hub
- **Purpose:** Model downloading
- **Details:** Models fetched via `transformers` / `huggingface-hub`; requires HF token for gated models

## Hardware / System

### NVIDIA NVML
- **Purpose:** GPU VRAM monitoring
- **Library:** `nvidia-ml-py` (13.590.48)
- **Details:** Direct NVML calls for memory usage tracking during inference

## HTTP

### httpx
- **Purpose:** Health check polling (e.g., waiting for vLLM server to be ready)
- **Details:** Async HTTP client used in test infrastructure

---
*Generated: 2026-03-11*
