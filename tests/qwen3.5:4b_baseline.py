import time
import torch
import wandb
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZES = [1, 2, 4, 8]
PROMPT_LENGTHS = [64, 128, 256]
MAX_NEW_TOKENS = 100
WARMUP_RUNS = 2

# --- Load ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
model.eval()

wandb.init(project="osso", name="baseline-tinyllama-static")

# --- Benchmark matrix ---
for batch_size in BATCH_SIZES:
    for prompt_len in PROMPT_LENGTHS:
        inputs = tokenizer("Explain attention mechanisms in transformers.", return_tensors="pt").input_ids
        inputs = inputs[:, :prompt_len].repeat(batch_size, 1).to("cuda")

        # Warmup
        for _ in range(WARMUP_RUNS):
            with torch.no_grad():
                model.generate(inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

        # TTFT
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        # Full generation
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0

        total_tokens = (outputs.shape[1] - inputs.shape[1]) * batch_size
        tpot_ms = ((total_s * 1000) - ttft_ms) / max(total_tokens - batch_size, 1)
        tokens_per_sec = total_tokens / total_s
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

        print(f"bs={batch_size} prompt={prompt_len} | ttft={ttft_ms:.1f}ms tpot={tpot_ms:.2f}ms tok/s={tokens_per_sec:.1f} vram={vram_peak_gb:.2f}GB")

        wandb.log({
            "batch_size": batch_size,
            "prompt_length": prompt_len,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "tokens_per_sec": tokens_per_sec,
            "vram_peak_gb": vram_peak_gb,
        })

# --- Profiler trace (single batch, 128 tokens) ---
inputs = tokenizer("Explain attention mechanisms.", return_tensors="pt").input_ids[:, :128].to("cuda")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    with record_function("generate"):
        with torch.no_grad():
            model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("baseline_trace.json")

wandb.finish()
