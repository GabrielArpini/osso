# Testing Patterns

**Analysis Date:** 2026-03-11

## Test Framework

**Runner:**
- No automated test framework (pytest/unittest not in dependencies)
- Tests executed as standalone scripts via direct execution

**Assertion Library:**
- No assertion library detected
- Manual validation through output inspection and wandb logging

**Run Commands:**
```bash
python tests/benchmark.py              # Run benchmark suite with multiprocessing
python tests/qwen3.5:4b_baseline.py    # Run baseline profiling
```

## Test Organization

**Location:**
- Tests separated in `tests/` directory
- Naming convention: descriptive names like `benchmark.py`, `qwen3.5:4b_baseline.py`
- No test discovery pattern (files must be run explicitly)

**Structure:**
```
tests/
├── benchmark.py              # Main benchmark harness with multiple backends
└── qwen3.5:4b_baseline.py    # Baseline profiling for TinyLlama model
```

## Test Types

**Benchmark Tests:**
- Files: `tests/benchmark.py`
- Scope: End-to-end inference performance measurement
- Approach:
  - Abstract `Backend` base class (`lines 89-101`) defines interface for different inference implementations
  - Concrete backends: `TransformersBackend`, `VLLMBackend`, `ServerBackend` base class
  - Parameters tested in matrix: batch sizes [1, 2, 4, 8], prompt lengths [64, 128, 256]
  - Metrics collected: TTFT (time-to-first-token), TPOT (time-per-output-token), tokens/sec, VRAM usage
  - Results logged to wandb and compared across backends

**Baseline Profiling:**
- File: `tests/qwen3.5:4b_baseline.py`
- Scope: Single-run profiling of TinyLlama model
- Approach:
  - Direct inference loops without abstraction
  - Torch profiler integration: `profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True)`
  - Chrome trace export for flamegraph analysis: `prof.export_chrome_trace("baseline_trace.json")`
  - wandb integration for metric logging

## Test Harness Design

**Benchmark Harness:**
- File: `tests/benchmark.py` (lines 257-366)
- Entry point: `if __name__ == "__main__":` (line 348)
- Process model:
  ```python
  multiprocessing.set_start_method("spawn")  # Full isolation per backend

  for backend_name in ["transformers", "vllm"]:
      p = multiprocessing.Process(target=_run_backend_process, args=(backend_name, os_idle_gb, result_queue))
      p.start()
      p.join()
  ```

**Setup & Teardown:**
- Abstract pattern via `Backend` class:
  - `setup()`: Initialize model/tokenizer, configure generation settings
  - `generate_batch()`: Execute inference with metrics collection
  - `teardown()`: Clean CUDA context, release memory
- Concrete example (TransformersBackend, lines 108-114):
  ```python
  def setup(self):
      self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
      self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
      self.model.eval()
      self.model.generation_config.max_length = None
  ```

## Test Instrumentation

**VRAM Monitoring:**
- Class: `VramMonitor` (lines 62-85)
- Approach: Background thread polling GPU memory at 50ms intervals
- Used in: `TransformersBackend.generate_batch()` and `ServerBackend.generate_batch()`
- Pattern:
  ```python
  monitor = VramMonitor()
  monitor.start()
  # ... inference code ...
  vram_peak_gb = monitor.stop()
  ```

**Timing:**
- `time.perf_counter()` for high-resolution measurements
- CUDA synchronization before/after critical sections: `torch.cuda.synchronize()`
- Warmup runs before actual benchmark: `WARMUP_RUNS = 2`

**GPU Monitoring (NVML):**
- File: `tests/benchmark.py` (lines 43-49)
- Initialization at module load: `pynvml.nvmlInit()`, `_nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)`
- Function: `_vram_used_gb()` - polls current GPU memory usage
- Function: `sample_vram_baseline()` - captures idle baseline over 5 seconds

## Metrics & Logging

**Wandb Integration:**
- Project: `osso`
- Run naming: `f"benchmark-{backend.name}-llama3.2-1b"` or `"baseline-tinyllama-static"`
- Metrics logged per configuration:
  ```python
  wandb.log({
      "batch_size": batch_size,
      "prompt_length": prompt_len,
      "ttft_ms": ttft_ms,
      "tpot_ms": tpot_ms,
      "tokens_per_sec": tokens_per_sec,
      "vram_peak_gb": vram_peak_gb,
  })
  ```

**Metrics Definitions:**
- TTFT: Time to First Token (ms) - latency of first output token
- TPOT: Time Per Output Token (ms) - average latency for subsequent tokens
- Tokens/sec: Throughput in tokens per second
- Model VRAM: GPU memory occupied by model weights alone
- Generation VRAM: Peak GPU memory during inference (model + activations + KV cache)

**Output Summary:**
- ASCII table generation via `_print_summary()` (lines 302-346)
- Box-drawing characters for formatting: `┌─┬┬┐` etc.
- Comparative display across backends

## Test Isolation

**Process Isolation:**
- Each backend runs in separate process: `multiprocessing.Process(target=_run_backend_process, ...)`
- `multiprocessing.set_start_method("spawn")` ensures full CUDA context separation
- Result queue for inter-process communication: `multiprocessing.Queue()`

**Resource Cleanup:**
- Server backends: `terminate()` and `wait()` on subprocess (lines 230-233)
- Transformers backend: `del self.model` and `torch.cuda.empty_cache()` (lines 153-154)
- No global state pollution between test runs

## Test Data

**Prompts:**
- Static prompts: `PROMPT_TEXT` (lines 25-40 in benchmark.py)
- Length variations tested: 64, 128, 256 tokens (variable length via tokenizer slicing)
- Model: Fixed to `MODEL_ID = "meta-llama/Llama-3.2-1B"` for benchmark suite

**Baseline Script:**
- Model: `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` (different from benchmark suite)
- Fixed prompt: `"Explain attention mechanisms in transformers."` (line 23)

## Profiling

**Torch Profiler:**
- File: `tests/qwen3.5:4b_baseline.py` (lines 68-74)
- Configuration:
  ```python
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
      with record_function("generate"):
          with torch.no_grad():
              model.generate(...)
  ```
- Output: Chrome trace for flamegraph analysis: `baseline_trace.json`

## Current Gaps

**No Unit Testing:**
- No isolated unit tests for individual components (`RMSNorm`, `RoPE`, `SwiGLU`)
- No mocking or fixtures for testing layers in isolation
- Testing limited to end-to-end inference benchmarks

**No Assertion Framework:**
- No automated pass/fail criteria
- Results validated manually via output inspection
- No CI/CD integration detected

**No Regression Testing:**
- No baseline comparisons stored
- Regression detection manual (comparing wandb runs)

---

*Testing analysis: 2026-03-11*
