# Coding Conventions

**Analysis Date:** 2026-03-11

## Naming Patterns

**Files:**
- Lower case with underscores: `layers.py`, `benchmark.py`, `llama.py`
- Test files placed in `tests/` directory without special suffix

**Functions:**
- Snake case: `_vram_used_gb()`, `sample_vram_baseline()`, `precompute_freqs_cis()`
- Private functions prefixed with single underscore: `_vram_used_gb()`, `_run()`, `_single_request()`
- Type hints used consistently for parameters and return values

**Variables:**
- Snake case for local variables and module constants
- Constants in ALL_CAPS: `MODEL_ID`, `VLLM_PYTHON`, `BATCH_SIZES`, `PROMPT_LENGTHS`, `MAX_NEW_TOKENS`
- Private instance variables prefixed with underscore: `self._interval`, `self._peak`, `self._stop`, `self._thread`, `self._port`, `self._launch_cmd`, `self._proc`, `self._client`

**Types & Classes:**
- PascalCase: `RMSNorm`, `SwiGLU`, `VramMonitor`, `Backend`, `TransformersBackend`, `ServerBackend`, `VLLMBackend`
- Classes extend base classes explicitly: `class RMSNorm(nn.Module):`, `class VramMonitor:`, `class Backend(abc.ABC):`

**Type Hints:**
- Comprehensive use of type hints throughout: `def _vram_used_gb() -> float:`, `def RoPE(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:`
- Using `from typing import` for complex types like `Tuple`

## Code Style

**Formatting:**
- No automated formatter detected (ruff/black not in dependencies)
- Consistent 4-space indentation observed
- Long lines preserved when readability allows (e.g., benchmark messages with 80+ characters)

**Linting:**
- No linting configuration detected in pyproject.toml
- No .flake8 or .pylintrc files present

**Spacing:**
- Class definitions separated by blank lines
- Methods within classes indented 4 spaces
- Docstrings used selectively (seen on `VramMonitor` and utility functions)

## Import Organization

**Order:**
1. Standard library imports: `abc`, `concurrent.futures`, `multiprocessing`, `os`, `subprocess`, `threading`, `time`
2. Third-party imports: `httpx`, `pynvml`, `torch`, `wandb`, `weave`, `torch.nn`, `torch.nn.functional`, `einops`
3. Transformers and specific model imports: `from transformers import AutoModelForCausalLM, AutoTokenizer`

**Path Aliases:**
- No path aliases detected (no import shortcuts like `from . import`)

**Examples:**
- `src/models/layers.py`: Imports organized by stdlib, torch packages, then einops
- `tests/benchmark.py`: Imports organized as stdlib → external packages → transformers

## Error Handling

**Patterns:**
- Broad exception catching observed: `except Exception:` in `_wait_ready()` method (line 175 of benchmark.py)
- No custom exception classes defined
- Timeout-based error handling: raises `TimeoutError` with descriptive message when server startup fails
- Missing token time handled with conditional: `ttft_ms = (first_token_t - t0) * 1000 if first_token_t else 0.0`

**Error Messages:**
- Include context: `raise TimeoutError(f"{self.name} server did not start within {timeout_s}s")`

## Logging

**Framework:** No centralized logging framework

**Patterns:**
- Print statements used for console output: `print(f"[{backend.name}] model VRAM: {model_vram_gb:.2f}GB...")`
- Logging integrated with wandb for metrics: `wandb.log({"model_vram_gb": model_vram_gb})`
- Progress reporting in benchmarks with formatted output
- ASCII table formatting for summary results (`_print_summary()` function uses box-drawing characters)

## Comments

**When to Comment:**
- Block comments for major sections: `# ── NVML ──────────────────────────────────────────────────────────────────────`
- Inline comments explain non-obvious logic: `# Python binary inside vLLM's isolated venv`
- Comment on data flow: `# TTFT`, `# Full generation`, `# Warmup`

**Documentation:**
- Docstrings provided for class/function purposes: `"""Sample VRAM for duration_s seconds and return the mean. Use at idle before loading anything."""`
- Method docstrings describe return values: `"""Returns (ttft_ms, token_count, total_s)."""`

## Function Design

**Size:**
- Functions kept reasonably small (10-50 lines typical)
- Longer functions group related operations: `generate_batch()` ~30 lines, `_print_summary()` ~35 lines

**Parameters:**
- Type hints on all parameters
- Default values for optional parameters: `sample_vram_baseline(duration_s: float = 5.0, interval_s: float = 0.05) -> float:`
- No *args or **kwargs pattern observed (direct parameter passing preferred)

**Return Values:**
- Single return type preferred
- Tuples used for multiple returns: `_single_request()` returns `tuple[float, int, float]`
- Dictionaries for complex metric collections: `return dict(ttft_ms=..., tpot_ms=..., tokens_per_sec=..., vram_peak_gb=...)`

**Docstrings:**
- Format: One-liner plus additional details when needed
- Placed immediately after function signature
- Include purpose and usage constraints

## Module Design

**Exports:**
- No `__all__` defined
- Public classes exported implicitly: `RMSNorm`, `SwiGLU`, `RoPE`, `precompute_freqs_cis`
- Backend classes form inheritance hierarchy accessible to consumers

**Class Structure:**
- Abstract base class pattern: `Backend(abc.ABC)` with `@abc.abstractmethod` decorators
- Concrete implementations inherit and implement interface: `TransformersBackend(Backend)`, `VLLMBackend(ServerBackend)`
- Initialization patterns vary:
  - Simple: `RMSNorm.__init__()` stores parameters
  - Complex: `VLLMBackend.__init__()` calls `super().__init__()` with launch configuration

**Module-Level Constants:**
- Configuration constants at top of files: `MODEL_ID`, `BATCH_SIZES`, `PROMPT_LENGTHS`, `MAX_NEW_TOKENS`, `WARMUP_RUNS`, `PROMPT_TEXT`
- VRAM monitoring initialized at module load: `pynvml.nvmlInit()`, `_nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)`

---

*Convention analysis: 2026-03-11*
