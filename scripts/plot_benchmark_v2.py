import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

LOG_NOKV = Path("/home/arpola/bench_nokv.log")
LOG_KV   = Path("/home/arpola/bench_kv.log")

SERIES   = ["nokv", "kv"]
COLORS   = {"nokv": "#8b949e", "kv": "#f85149"}
LABELS   = {
    "en": {"nokv": "No KV cache", "kv": "With KV cache"},
    "pt": {"nokv": "Sem KV cache", "kv": "Com KV cache"},
}

ROW_RE = re.compile(
    r"^\[osso\]\s+bs=(\d+)\s+prompt=(\d+)\s+\|\s+ttft=([\d.]+)ms\s+tpot=([\d.]+)ms\s+tok/s=([\d.]+)"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        rows.append({
            "bs": int(m.group(1)),
            "prompt": int(m.group(2)),
            "ttft": float(m.group(3)),
            "tpot": float(m.group(4)),
            "tokps": float(m.group(5)),
        })
    return rows


def by_prompt(rows: list[dict], bs: int, col: str, prompts: list[int]) -> list[float]:
    lookup = {r["prompt"]: r[col] for r in rows if r["bs"] == bs}
    return [lookup[p] for p in prompts]


data = {"nokv": parse_log(LOG_NOKV), "kv": parse_log(LOG_KV)}
PROMPT_LENS = sorted({r["prompt"] for r in data["kv"]})

BG, PANEL_BG, GRID_C = "#0d1117", "#161b22", "#21262d"
TEXT, SUBTEXT = "#e6edf3", "#8b949e"

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

STRINGS = {
    "en": {
        "title":       "osso inference engine — iteration #2",
        "subtitle":    "Llama 3.2 1B · RTX 2060 6 GB · naïve KV cache",
        "tpot_title":  "TPOT by prompt length\n(batch=1)",
        "tpot_xlabel": "input tokens",
        "tpot_ylabel": "ms / output token",
        "ttft_title":  "TTFT by prompt length\n(batch=1)",
        "ttft_xlabel": "input tokens",
        "ttft_ylabel": "ms to first token",
        "tput_title":  "Throughput by prompt length\n(batch=1)",
        "tput_xlabel": "input tokens",
        "tput_ylabel": "tokens / second",
        "footer":      "TPOT = time per output token  ·  TTFT = time to first token",
        "github":      "github.com/GabrielArpini/osso",
    },
    "pt": {
        "title":       "osso inference engine — iteração #2",
        "subtitle":    "Llama 3.2 1B · RTX 2060 6 GB · cache KV ingênuo",
        "tpot_title":  "TPOT por tamanho de prompt\n(batch=1)",
        "tpot_xlabel": "tokens de entrada",
        "tpot_ylabel": "ms / token gerado",
        "ttft_title":  "TTFT por tamanho de prompt\n(batch=1)",
        "ttft_xlabel": "tokens de entrada",
        "ttft_ylabel": "ms até 1º token",
        "tput_title":  "Throughput por tamanho de prompt\n(batch=1)",
        "tput_xlabel": "tokens de entrada",
        "tput_ylabel": "tokens / segundo",
        "footer":      "TPOT = tempo por token gerado  ·  TTFT = tempo até o 1º token",
        "github":      "github.com/GabrielArpini/osso",
    },
}


def build_figure(lang: str) -> plt.Figure:
    s = STRINGS[lang]
    plt.rcParams.update(BASE_PARAMS)

    fig = plt.figure(figsize=(18, 9), dpi=150)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97,
        top=0.76, bottom=0.12,
        hspace=0.55, wspace=0.35,
    )

    fig.text(0.06, 0.93, s["title"], fontsize=22, fontweight="bold", color=TEXT, va="top")
    fig.text(0.06, 0.865, s["subtitle"], fontsize=11, color=SUBTEXT, va="top")

    for i, key in enumerate(SERIES):
        fig.text(0.70 + i * 0.14, 0.93, f"● {LABELS[lang][key]}",
                 fontsize=10, color=COLORS[key], va="top", fontweight="bold")

    panels = [
        ("tpot",  "tpot",  s["tpot_title"], s["tpot_xlabel"], s["tpot_ylabel"]),
        ("ttft",  "ttft",  s["ttft_title"], s["ttft_xlabel"], s["ttft_ylabel"]),
        ("tokps", "tokps", s["tput_title"], s["tput_xlabel"], s["tput_ylabel"]),
    ]
    for i, (_, col, title, xlab, ylab) in enumerate(panels):
        ax = fig.add_subplot(gs[:, i])
        ax.set_facecolor(PANEL_BG)
        ax.grid(axis="y", linestyle="--")
        for key in SERIES:
            ax.plot(PROMPT_LENS, by_prompt(data[key], 1, col, PROMPT_LENS),
                    color=COLORS[key], linewidth=2.5,
                    marker="o", markersize=7, markerfacecolor=BG,
                    markeredgewidth=2.5, zorder=3)
        ax.set_title(title, color=TEXT, fontsize=11, pad=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xticks(PROMPT_LENS)
        ax.tick_params(length=0)

    fig.text(0.06, 0.07, s["footer"], fontsize=8, color=SUBTEXT, va="top")
    fig.text(0.97, 0.07, s["github"], fontsize=8, color=SUBTEXT, va="top", ha="right")

    return fig


for lang in ["en", "pt"]:
    fig = build_figure(lang)
    out = f"imgs/benchmark_v2_{lang}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved -> {out}")
