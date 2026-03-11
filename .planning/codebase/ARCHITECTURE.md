# Architecture

**Analysis Date:** 2026-03-11

## Pattern Overview

**Overall:** Modular neural network layer library with benchmarking framework

**Key Characteristics:**
- Layer-based composable architecture for transformer models
- Inference-focused with low VRAM constraints (target: 6GB VRAM on Turing GPUs)
- Benchmarking harness comparing multiple inference backends
- Direct PyTorch nn.Module implementations without abstraction layers

## Layers

**Model Layer:**
- Purpose: Define trainable Llama model architecture
- Location: `models/llama.py`
- Contains: Llama 3.2 model definitions
- Depends on: PyTorch, layers module
- Used by: Test harnesses and benchmarks

**Primitive Layers:**
- Purpose: Implement optimized transformer components (RMSNorm, RoPE, SwiGLU)
- Location: `models/layers.py`
- Contains: Custom implementations of normalization, positional embeddings, and feed-forward layers
- Depends on: PyTorch, einops for tensor manipulation
- Used by: Model layer, benchmarks

**Benchmark Framework:**
- Purpose: Compare inference performance across backends (Transformers, vLLM)
- Location: `tests/benchmark.py`
- Contains: Abstract Backend interface, concrete implementations, performance monitoring
- Depends on: PyTorch, transformers library, vLLM, OpenAI client, NVML for GPU monitoring
- Used by: Performance validation and profiling

**Baseline Tests:**
- Purpose: Establish performance baselines for reference models
- Location: `tests/qwen3.5:4b_baseline.py`
- Contains: Simple benchmark harness for TinyLlama model
- Depends on: PyTorch, transformers, Weights & Biases
- Used by: Performance comparison and regression detection

## Data Flow

**Inference Pipeline (Transformer Backend):**

1. Tokenize input text into token IDs using AutoTokenizer
2. Load model with `AutoModelForCausalLM.from_pretrained()` to CUDA
3. Batch input tokens and move to GPU
4. Run warmup iterations (2x 10-token generation)
5. Measure Time-To-First-Token (TTFT) - single forward pass + one decoding step
6. Measure Time-Per-Output-Token (TPOT) - remaining decoding steps amortized
7. Measure peak VRAM during full generation
8. Compute tokens/sec throughput metric
9. Log metrics to Weights & Biases

**Inference Pipeline (Server Backend - vLLM):**

1. Launch vLLM server subprocess on port 30001
2. Wait for server readiness (/health check)
3. Use OpenAI-compatible client to send requests
4. Stream token responses for TTFT measurement
5. Measure concurrent batch requests via ThreadPoolExecutor
6. Monitor peak VRAM via background thread polling
7. Log aggregated metrics to Weights & Biases
8. Terminate server process on teardown

**State Management:**
- Model state: Loaded into GPU memory, no persistent state between runs
- Benchmark state: In-memory per-backend, isolated via multiprocessing (spawn method)
- Metrics state: Collected in-memory, written to W&B cloud
- GPU memory: Explicitly cleared between backends via `torch.cuda.empty_cache()`

## Key Abstractions

**Backend Interface:**
- Purpose: Define common contract for inference implementations
- Examples: `TransformersBackend`, `ServerBackend`, `VLLMBackend`
- Pattern: Abstract base class with `setup()`, `generate_batch()`, `teardown()` lifecycle
- Enables: Easy addition of new inference engines, comparable metrics

**VramMonitor:**
- Purpose: Capture peak GPU memory usage during generation
- Examples: `models/layers.py` - monitors during inference
- Pattern: Background thread polling VRAM status with configurable interval
- Enables: Accurate peak memory measurement without language profiler overhead

**Benchmark Metrics:**
- Purpose: Normalize performance across backends
- Metrics: TTFT (Time-To-First-Token), TPOT (Time-Per-Output-Token), tokens/sec, VRAM
- Pattern: Fixed dictionary return format from `generate_batch()`
- Enables: Deterministic comparison via `_print_summary()`

## Entry Points

**Benchmark Harness:**
- Location: `tests/benchmark.py`
- Triggers: `python tests/benchmark.py`
- Responsibilities:
  - Spawn child processes for TransformersBackend and VLLMBackend
  - Collect metrics across batch sizes [1, 2, 4, 8] and prompt lengths [64, 128, 256]
  - Print comparative summary table
  - Log all results to Weights & Biases

**Baseline Test:**
- Location: `tests/qwen3.5:4b_baseline.py`
- Triggers: Direct Python execution for reference model benchmarking
- Responsibilities:
  - Load TinyLlama model and measure performance
  - Generate chrome trace for profiling
  - Log baseline metrics to W&B

## Error Handling

**Strategy:** Defensive timeout-based approach for server readiness

**Patterns:**
- Server startup waits 360s with 3s polling interval, raises TimeoutError if not ready
- Background VRAM monitoring captures exceptions silently (expected if GPU is reset)
- Subprocess termination via `.terminate()` + `.wait()` with no timeout enforcement
- Missing server endpoints fail gracefully (exception caught, retry loop continues)

## Cross-Cutting Concerns

**Logging:**
- Method: Direct `print()` statements with structured output
- Format: `[backend_name] metric=value` for consistency
- W&B integration: All metrics logged to Weights & Biases project "osso"

**Validation:**
- Input validation: Tokenizer accepts string prompt, outputs tensor
- Model validation: `model.eval()` sets inference mode, no dropout
- Numeric validation: TTFT/TPOT computed with `max(..., 1)` to prevent division by zero

**Synchronization:**
- CUDA synchronization: Explicit `torch.cuda.synchronize()` before/after timed sections
- Thread synchronization: Background monitor uses `threading.Event()` for clean shutdown
- Process synchronization: Main process joins spawned backends via `multiprocessing.Process.join()`

---

*Architecture analysis: 2026-03-11*
