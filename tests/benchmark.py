import abc
import concurrent.futures
import multiprocessing
import os
import subprocess
import threading
import time

import httpx
import pynvml
import torch
import weave
import wandb
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"

# Python binary inside vLLM's isolated venv
VLLM_PYTHON = os.path.expanduser("~/.local/share/vllm-venv/bin/python")
BATCH_SIZES = [1, 2, 4, 8]
PROMPT_LENGTHS = [64, 128, 256]
MAX_NEW_TOKENS = 100
WARMUP_RUNS = 2
PROMPT_TEXT = (
    "Attention mechanisms in transformer models work by allowing each token in a sequence to "
    "attend to every other token, computing a weighted sum of value vectors where the weights "
    "are determined by the compatibility between query and key vectors. The scaled dot-product "
    "attention computes scores by taking the dot product of queries and keys, scaling by the "
    "square root of the key dimension to prevent vanishing gradients, then applying a softmax "
    "to obtain a probability distribution over positions. Multi-head attention extends this by "
    "running several attention operations in parallel with different learned projections, "
    "allowing the model to jointly attend to information from different representation subspaces. "
    "The self-attention mechanism enables transformers to capture long-range dependencies without "
    "the sequential bottleneck of recurrent networks, making parallelization during training "
    "straightforward. Positional encodings are added to the input embeddings to inject information "
    "about the relative or absolute position of tokens in the sequence, since the attention "
    "operation itself is permutation-invariant. Cross-attention in encoder-decoder architectures "
    "allows the decoder to attend to the encoder output, conditioning generation on the full "
    "input context at every decoding step."
)

# ── NVML ──────────────────────────────────────────────────────────────────────
pynvml.nvmlInit()
_nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def _vram_used_gb() -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / 1024**3


def sample_vram_baseline(duration_s: float = 5.0, interval_s: float = 0.05) -> float:
    """Sample VRAM for duration_s seconds and return the mean. Use at idle before loading anything."""
    samples = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        samples.append(_vram_used_gb())
        time.sleep(interval_s)
    return sum(samples) / len(samples)


class VramMonitor:
    """Polls GPU memory in a background thread to capture peak."""

    def __init__(self, interval_s: float = 0.05):
        self._interval = interval_s
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._peak = _vram_used_gb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, _vram_used_gb())
            time.sleep(self._interval)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self._peak


# ── Abstract backend ──────────────────────────────────────────────────────────
class Backend(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def setup(self): ...

    @abc.abstractmethod
    def generate_batch(self, prompt_text: str, batch_size: int, max_new_tokens: int) -> dict: ...

    @abc.abstractmethod
    def teardown(self): ...


# ── Transformers (osso baseline) ──────────────────────────────────────────────
class TransformersBackend(Backend):
    name = "transformers"

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float16, device_map="cuda"
        )
        self.model.eval()
        self.model.generation_config.max_length = None

    def generate_batch(self, prompt_text: str, batch_size: int, max_new_tokens: int) -> dict:
        enc = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc.input_ids.repeat(batch_size, 1).to("cuda")
        attention_mask = enc.attention_mask.repeat(batch_size, 1).to("cuda")
        gen_kwargs = dict(attention_mask=attention_mask, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)

        for _ in range(WARMUP_RUNS):
            with torch.no_grad():
                self.model.generate(input_ids, max_new_tokens=10, **gen_kwargs)
        torch.cuda.synchronize()

        # TTFT
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model.generate(input_ids, max_new_tokens=1, **gen_kwargs)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        # Full generation
        monitor = VramMonitor()
        monitor.start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
        torch.cuda.synchronize()
        total_s = time.perf_counter() - t0
        vram_peak_gb = monitor.stop()

        total_tokens = (outputs.shape[1] - input_ids.shape[1]) * batch_size
        tpot_ms = ((total_s * 1000) - ttft_ms) / max(total_tokens - batch_size, 1)
        tokens_per_sec = total_tokens / total_s

        return dict(ttft_ms=ttft_ms, tpot_ms=tpot_ms, tokens_per_sec=tokens_per_sec, vram_peak_gb=vram_peak_gb)

    def teardown(self):
        del self.model
        torch.cuda.empty_cache()


# ── Server-based backend base ─────────────────────────────────────────────────
class ServerBackend(Backend):
    def __init__(self, port: int, launch_cmd: list[str]):
        self._port = port
        self._launch_cmd = launch_cmd
        self._proc = None
        self._client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")

    def setup(self):
        self._proc = subprocess.Popen(self._launch_cmd)
        self._wait_ready()

    def _wait_ready(self, timeout_s: int = 360):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                httpx.get(f"http://localhost:{self._port}/health", timeout=2)
                return
            except Exception:
                time.sleep(3)
        raise TimeoutError(f"{self.name} server did not start within {timeout_s}s")

    def _single_request(self, prompt_text: str, max_tokens: int) -> tuple[float, int, float]:
        """Returns (ttft_ms, token_count, total_s)."""
        t0 = time.perf_counter()
        resp = self._client.completions.create(
            model=MODEL_ID,
            prompt=prompt_text,
            max_tokens=max_tokens,
            stream=True,
            temperature=0,
        )
        first_token_t = None
        token_count = 0
        for chunk in resp:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            if chunk.choices[0].text:
                token_count += 1
        total_s = time.perf_counter() - t0
        ttft_ms = (first_token_t - t0) * 1000 if first_token_t else 0.0
        return ttft_ms, token_count, total_s

    def generate_batch(self, prompt_text: str, batch_size: int, max_new_tokens: int) -> dict:
        # Warmup
        for _ in range(WARMUP_RUNS):
            self._client.completions.create(
                model=MODEL_ID, prompt=prompt_text, max_tokens=10, temperature=0
            )

        # TTFT probe — single request, 1 token
        ttft_ms, _, _ = self._single_request(prompt_text, max_tokens=1)

        # Full batch — concurrent requests
        monitor = VramMonitor()
        monitor.start()
        t_batch_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                pool.submit(self._single_request, prompt_text, max_new_tokens)
                for _ in range(batch_size)
            ]
            results = [f.result() for f in futures]
        total_batch_s = time.perf_counter() - t_batch_start
        vram_peak_gb = monitor.stop()

        total_tokens = sum(r[1] for r in results)
        tokens_per_sec = total_tokens / total_batch_s
        tpot_ms = ((total_batch_s * 1000) - ttft_ms) / max(total_tokens - batch_size, 1)

        return dict(ttft_ms=ttft_ms, tpot_ms=tpot_ms, tokens_per_sec=tokens_per_sec, vram_peak_gb=vram_peak_gb)

    def teardown(self):
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None


# ── vLLM ──────────────────────────────────────────────────────────────────────
class VLLMBackend(ServerBackend):
    name = "vllm"

    def __init__(self):
        super().__init__(
            port=30001,
            launch_cmd=[
                VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_ID,
                "--host", "0.0.0.0",
                "--port", "30001",
                "--gpu-memory-utilization", "0.50",
                "--max-model-len", "1024",
                "--max-num-seqs", "32",
                "--enforce-eager",
            ],
        )


# ── Benchmark harness ─────────────────────────────────────────────────────────
def run_benchmark(backend: Backend, tokenizer: AutoTokenizer, os_idle_gb: float, result_queue=None):
    wandb.init(project="osso", name=f"benchmark-{backend.name}-llama3.2-1b", reinit=True)

    backend.setup()
    model_loaded_gb = sample_vram_baseline(duration_s=3.0)
    model_vram_gb = model_loaded_gb - os_idle_gb

    print(f"[{backend.name}] model VRAM: {model_vram_gb:.2f}GB (idle baseline: {os_idle_gb:.2f}GB)")
    wandb.log({"model_vram_gb": model_vram_gb})

    collected = []
    for batch_size in BATCH_SIZES:
        for prompt_len in PROMPT_LENGTHS:
            ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[:, :prompt_len]
            prompt_text = tokenizer.decode(ids[0], skip_special_tokens=True)

            metrics = backend.generate_batch(prompt_text, batch_size, MAX_NEW_TOKENS)
            metrics["generation_vram_gb"] = metrics.pop("vram_peak_gb") - model_loaded_gb
            metrics["model_vram_gb"] = model_vram_gb

            print(
                f"[{backend.name}] bs={batch_size} prompt={prompt_len} | "
                f"ttft={metrics['ttft_ms']:.1f}ms tpot={metrics['tpot_ms']:.2f}ms "
                f"tok/s={metrics['tokens_per_sec']:.1f} gen_vram={metrics['generation_vram_gb']:.2f}GB"
            )
            wandb.log({"batch_size": batch_size, "prompt_length": prompt_len, **metrics})
            collected.append((batch_size, prompt_len, metrics))

    backend.teardown()
    wandb.finish()

    if result_queue is not None:
        result_queue.put((backend.name, collected))


def _run_backend_process(backend_name: str, os_idle_gb: float, result_queue):
    """Runs in a spawned child process — CUDA context fully released on exit."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    backend = {
        "transformers": TransformersBackend,
        "vllm": VLLMBackend,
    }[backend_name]()
    run_benchmark(backend, tokenizer, os_idle_gb, result_queue)


def _print_summary(all_results: dict):
    backends = list(all_results.keys())
    if len(backends) < 2:
        return

    b0, b1 = backends[0], backends[1]
    data0 = {(bs, pl): m for bs, pl, m in all_results[b0]}
    data1 = {(bs, pl): m for bs, pl, m in all_results[b1]}
    keys = sorted(set(data0) & set(data1))

    lw, cw = 20, 16

    top    = f"┌{'─'*(lw+2)}┬{'─'*(cw+2)}┬{'─'*(cw+2)}┐"
    head   = f"│ {'BENCHMARK SUMMARY':<{lw}} │ {b0:^{cw}} │ {b1:^{cw}} │"
    mid    = f"├{'─'*(lw+2)}┼{'─'*(cw+2)}┼{'─'*(cw+2)}┤"
    bot    = f"└{'─'*(lw+2)}┴{'─'*(cw+2)}┴{'─'*(cw+2)}┘"

    def row(label, v0, v1):
        return f"│ {label:<{lw}} │ {v0:>{cw}} │ {v1:>{cw}} │"

    span_w = lw + 2 + 1 + cw + 2 + 1 + cw + 2
    def span(label):
        return f"│ {label:<{span_w - 2}} │"

    print()
    print(top)
    print(head)

    for bs, pl in keys:
        m0, m1 = data0[(bs, pl)], data1[(bs, pl)]
        print(mid)
        print(span(f"bs={bs}  prompt={pl}"))
        print(mid)
        print(row("  ttft (ms)",    f"{m0['ttft_ms']:.1f}",        f"{m1['ttft_ms']:.1f}"))
        print(mid)
        print(row("  tpot (ms)",    f"{m0['tpot_ms']:.2f}",        f"{m1['tpot_ms']:.2f}"))
        print(mid)
        print(row("  tok/s",        f"{m0['tokens_per_sec']:.1f}", f"{m1['tokens_per_sec']:.1f}"))

    vram0 = all_results[b0][0][2]["model_vram_gb"]
    vram1 = all_results[b1][0][2]["model_vram_gb"]
    print(mid)
    print(row("model vram (GB)", f"{vram0:.2f}", f"{vram1:.2f}"))
    print(bot)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    print("Sampling OS idle VRAM baseline for 5s...")
    os_idle_gb = sample_vram_baseline(duration_s=5.0)
    print(f"OS idle baseline: {os_idle_gb:.2f}GB")

    result_queue = multiprocessing.Queue()
    all_results = {}

    for backend_name in ["transformers", "vllm"]:
        p = multiprocessing.Process(target=_run_backend_process, args=(backend_name, os_idle_gb, result_queue))
        p.start()
        p.join()
        if not result_queue.empty():
            name, collected = result_queue.get()
            all_results[name] = collected

    _print_summary(all_results)
