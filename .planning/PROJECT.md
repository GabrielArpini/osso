# osso

## What This Is

osso is a bare-bones LLM inference engine built from scratch, optimized for low-VRAM consumer GPUs (target: RTX 2060 6GB, Turing/SM75). It implements the Llama architecture natively, builds custom Triton kernels to work around CUDA feature gaps on older hardware (no flash attention v2), and serves as both a working inference engine and a platform for experimenting with kernel optimizations like fused operations, KV cache management, and quantization.

## Core Value

A Llama model runs end-to-end on a 6GB GPU, faster than naive PyTorch — proving that careful memory management and custom kernels can make older hardware viable for local LLM inference.

## Requirements

### Validated

- ✓ RMSNorm layer (PyTorch) — existing
- ✓ RoPE positional embeddings with einops — existing
- ✓ SwiGLU FFN layer — existing
- ✓ Benchmark harness for comparing inference backends — existing

### Active

- [ ] Full Llama 3.2 1B model architecture (attention, transformer blocks, LM head)
- [ ] HuggingFace weight loading into osso model
- [ ] KV cache with memory management (paged or pooled)
- [ ] Flash attention workaround for Turing GPUs (Triton, SM75-compatible)
- [ ] Fused Triton kernels (RMSNorm, SwiGLU, attention ops)
- [ ] int8/int4 quantization (self-implemented, not GGUF)
- [ ] Serving API (HTTP, format TBD)
- [ ] Extensible architecture registry (Llama-first, other models pluggable)

### Out of Scope

- GGUF / pre-quantized weight loading — implementing quantization ourselves
- Training — inference only
- Multi-GPU — single 6GB GPU is the constraint
- Mobile / edge deployment — desktop GPU focus

## Context

- GPU target: RTX 2060 (Turing, SM75, 6GB VRAM, no BF16 hardware support, no flash attention v2)
- Architecture borrowed from mini-sglang: scheduler + runtime + model runner structure
- Triton-first for custom kernels; raw CUDA (with CuTe/CUTLASS) as fallback for ops Triton can't express
- Starting with Llama 3.2 1B — fits comfortably in 6GB without quantization pressure, making the kernel work the variable
- Benchmarking harness already exists for comparing osso against HuggingFace transformers baseline
- Wandb integration already present for performance tracking

## Constraints

- **Hardware**: RTX 2060 (SM75) — no flash attention v2, limited SRAM, no BF16 hw support
- **VRAM**: 6GB hard ceiling — model + KV cache + activations must fit
- **Language**: Python + Triton (kernel code), PyTorch for non-performance-critical paths
- **Scope**: Single model serving, not a production system

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Triton-first for custom kernels | Easier than raw CUDA, sufficient for SM75 workarounds | — Pending |
| Start with Llama 3.2 1B | Small enough to fit without quantization — isolates kernel perf | — Pending |
| Self-implement quantization | Learning goal + better control over memory layout | — Pending |
| mini-sglang architecture | Proven structure for an inference engine, well understood | — Pending |

---
*Last updated: 2026-03-11 after initialization*
