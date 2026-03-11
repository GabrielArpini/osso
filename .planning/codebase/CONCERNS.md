# Concerns

## Incomplete Implementation

### `models/llama.py` is a stub
- **File:** `models/llama.py`
- **Details:** Only contains `import torch`. No model class, no forward pass, no weight loading.
- **Impact:** Core deliverable of the project does not exist yet.

### Missing attention mechanism
- **File:** `models/layers.py`
- **Details:** `RoPE` (positional encoding) and `SwiGLU` (FFN) are implemented, but there is no attention class. RoPE has no consumer yet.
- **Impact:** Cannot assemble a transformer block until attention is added.

## Bugs

### RMSNorm computes wrong norm for batched inputs
- **File:** `models/layers.py:16`
- **Details:** `torch.sum(torch.pow(x, 2))` sums over **all** elements in the tensor, then divides by `self.size` (feature dim only). For inputs with batch or sequence dimensions this produces a single scalar norm shared across the whole batch instead of a per-token norm.
- **Correct approach:** `torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)` — mean over the last dimension only.

## Code Quality

### Two benchmark scripts with unclear relationship
- **Files:** `tests/benchmark.py`, `tests/qwen3.5:4b_baseline.py`
- **Details:** `benchmark.py` is a full multi-backend harness (366 lines); `qwen3.5:4b_baseline.py` is a 76-line standalone script. No clear delineation of when to use which.

### Hardcoded vLLM virtualenv path
- **File:** `tests/benchmark.py`
- **Details:** Path to vLLM virtualenv is hardcoded. Breaks on any machine other than the original dev environment.

## Testing

### No unit tests for layer implementations
- **Details:** `tests/` contains only benchmark scripts. `RMSNorm`, `RoPE`, and `SwiGLU` have zero test coverage. The RMSNorm bug above would be caught immediately by a basic shape test.

## Infrastructure

### VramMonitor threading
- **File:** `tests/benchmark.py`
- **Details:** Background thread polls VRAM without explicit synchronization. Low risk for single-run benchmarks but could produce inconsistent readings under concurrent workloads.

---
*Generated: 2026-03-11*
