# Structure

## Directory Layout

```
osso/
├── models/
│   ├── llama.py          # Llama model class (currently empty — 3 lines, import only)
│   └── layers.py         # Layer primitives: RMSNorm, RoPE, SwiGLU (47 lines)
├── tests/
│   ├── benchmark.py      # Multi-backend benchmarking harness (366 lines)
│   └── qwen3.5:4b_baseline.py  # TinyLlama reference baseline (76 lines)
├── pyproject.toml        # Project metadata and dependencies (uv)
├── uv.lock               # Locked dependency versions
├── .python-version       # Python version pin (3.12)
└── README.md             # Project description
```

## Key Locations

| Path | Purpose |
|------|---------|
| `models/layers.py` | Reusable transformer layer components |
| `models/llama.py` | Llama model assembly (stub — not implemented) |
| `tests/benchmark.py` | Main benchmarking entry point |
| `tests/qwen3.5:4b_baseline.py` | Baseline comparison script |
| `pyproject.toml` | Dependency management via `uv` |

## Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `RMSNorm`, `RotaryEmbedding`, `SwiGLU`)
- **Functions/methods:** `snake_case`
- **Test files:** live in `tests/`, named by model or function being tested

## Module Organization

- `models/` — model definitions and layer components
- `tests/` — benchmarking and evaluation scripts (not unit tests)
- No `src/` layout — flat structure at repo root

---
*Generated: 2026-03-11*
