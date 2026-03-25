import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT  = "osso"
BACKENDS = ["transformers", "vllm", "osso"]

COLORS = {"transformers": "#58a6ff", "vllm": "#3fb950", "osso": "#f85149"}
LABELS = {"transformers": "Transformers", "vllm": "vLLM", "osso": "Osso"}

# ── Pull data from wandb ──────────────────────────────────────────────────────
def fetch_runs(project: str) -> dict:
    api = wandb.Api()
    result = {}
    for backend in BACKENDS:
        runs = api.runs(project, filters={"display_name": f"benchmark-{backend}-llama3.2-1b"})
        if not runs:
            raise RuntimeError(f"no wandb run found for {backend}")
        df = runs[0].history()
        result[backend] = df.dropna(subset=["batch_size"]).copy()
    return result

def metric_bs1(runs: dict, col: str) -> dict:
    out = {}
    for backend, df in runs.items():
        subset = df[df["batch_size"] == 1].sort_values("prompt_length")
        out[backend] = subset[col].tolist()
    return out

def metric_by_batch(runs: dict, col: str, prompt_len: int) -> dict:
    out = {}
    for backend, df in runs.items():
        subset = df[df["prompt_length"] == prompt_len].sort_values("batch_size")
        out[backend] = subset[col].tolist()
    return out

def prompt_lens(runs: dict) -> list:
    df = next(iter(runs.values()))
    return sorted(df["prompt_length"].unique().astype(int).tolist())

def batch_sizes(runs: dict) -> list:
    df = next(iter(runs.values()))
    return sorted(df["batch_size"].unique().astype(int).tolist())

print("fetching wandb runs...")
runs = fetch_runs(PROJECT)

PROMPT_LENS  = prompt_lens(runs)
BATCH_SIZES  = batch_sizes(runs)
tpot_bs1     = metric_bs1(runs, "tpot_ms")
ttft_bs1     = metric_bs1(runs, "ttft_ms")
tokps_p64    = metric_by_batch(runs, "tokens_per_sec", prompt_len=64)

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_C   = "#21262d"
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"

BASE_PARAMS = {
    "font.family":        "monospace",
    "text.color":         TEXT,
    "axes.facecolor":     PANEL_BG,
    "figure.facecolor":   BG,
    "axes.edgecolor":     GRID_C,
    "axes.labelcolor":    SUBTEXT,
    "xtick.color":        SUBTEXT,
    "ytick.color":        SUBTEXT,
    "grid.color":         GRID_C,
    "grid.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
}


def build_figure(lang: str) -> plt.Figure:
    is_pt = lang == "pt"

    strings = {
        "title":       "osso inference engine — iteration #1",
        "subtitle":    "Llama 3.2 1B · RTX 2060 6 GB · no optimizations",
        "tpot_title":  "TPOT by prompt length\n(batch=1)",
        "tpot_xlabel": "input tokens",
        "tpot_ylabel": "ms / output token",
        "ttft_title":  "TTFT by prompt length\n(batch=1)",
        "ttft_xlabel": "input tokens",
        "ttft_ylabel": "ms to first token",
        "tput_title":  "Throughput by batch size\n(prompt=64)",
        "tput_xlabel": "batch size",
        "tput_ylabel": "tokens / second",
        "footer":      "TPOT = time per output token  ·  TTFT = time to first token",
        "github":      "github.com/GabrielArpini/osso",
    } if not is_pt else {
        "title":       "osso inference engine — iteração #1",
        "subtitle":    "Llama 3.2 1B · RTX 2060 6 GB · sem otimizações",
        "tpot_title":  "TPOT por tamanho de prompt\n(batch=1)",
        "tpot_xlabel": "tokens de entrada",
        "tpot_ylabel": "ms / token gerado",
        "ttft_title":  "TTFT por tamanho de prompt\n(batch=1)",
        "ttft_xlabel": "tokens de entrada",
        "ttft_ylabel": "ms até 1º token",
        "tput_title":  "Throughput por batch size\n(prompt=64)",
        "tput_xlabel": "batch size",
        "tput_ylabel": "tokens / segundo",
        "footer":      "TPOT = tempo por token gerado  ·  TTFT = tempo até o 1º token",
        "github":      "github.com/GabrielArpini/osso",
    }

    plt.rcParams.update(BASE_PARAMS)

    fig = plt.figure(figsize=(18, 9), dpi=150)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97,
        top=0.76, bottom=0.12,
        hspace=0.55, wspace=0.35,
    )

    fig.text(0.06, 0.93, strings["title"], fontsize=22, fontweight="bold", color=TEXT, va="top")
    fig.text(0.06, 0.865, strings["subtitle"], fontsize=11, color=SUBTEXT, va="top")

    for i, (key, label) in enumerate(LABELS.items()):
        fig.text(0.63 + i * 0.12, 0.93, f"● {label}", fontsize=10, color=COLORS[key], va="top", fontweight="bold")

    # Panel 1: TPOT vs prompt length
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor(PANEL_BG)
    ax1.grid(axis="y", linestyle="--")
    for key in BACKENDS:
        ax1.plot(PROMPT_LENS, tpot_bs1[key], color=COLORS[key], linewidth=2.5,
                 marker="o", markersize=7, markerfacecolor=BG, markeredgewidth=2.5, zorder=3)
    ax1.set_title(strings["tpot_title"], color=TEXT, fontsize=11, pad=10)
    ax1.set_xlabel(strings["tpot_xlabel"])
    ax1.set_ylabel(strings["tpot_ylabel"])
    ax1.set_xticks(PROMPT_LENS)
    ax1.tick_params(length=0)

    # Panel 2: TTFT vs prompt length
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.set_facecolor(PANEL_BG)
    ax2.grid(axis="y", linestyle="--")
    for key in BACKENDS:
        ax2.plot(PROMPT_LENS, ttft_bs1[key], color=COLORS[key], linewidth=2.5,
                 marker="o", markersize=7, markerfacecolor=BG, markeredgewidth=2.5, zorder=3)
    ax2.set_title(strings["ttft_title"], color=TEXT, fontsize=11, pad=10)
    ax2.set_xlabel(strings["ttft_xlabel"])
    ax2.set_ylabel(strings["ttft_ylabel"])
    ax2.set_xticks(PROMPT_LENS)
    ax2.tick_params(length=0)

    # Panel 3: Throughput vs batch size
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.set_facecolor(PANEL_BG)
    ax3.grid(axis="y", linestyle="--")
    w = 0.22
    x = np.arange(len(BATCH_SIZES))
    for i, key in enumerate(BACKENDS):
        ax3.bar(x + (i - 1) * w, tokps_p64[key], width=w,
                color=COLORS[key], alpha=0.85, zorder=3)
    ax3.set_title(strings["tput_title"], color=TEXT, fontsize=11, pad=10)
    ax3.set_xlabel(strings["tput_xlabel"])
    ax3.set_ylabel(strings["tput_ylabel"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(BATCH_SIZES)
    ax3.tick_params(length=0)

    fig.text(0.06, 0.07, strings["footer"], fontsize=8, color=SUBTEXT, va="top")
    fig.text(0.97, 0.07, strings["github"], fontsize=8, color=SUBTEXT, va="top", ha="right")

    return fig


for lang in ["en", "pt"]:
    fig = build_figure(lang)
    out = f"imgs/benchmark_v1_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved → {out}")
