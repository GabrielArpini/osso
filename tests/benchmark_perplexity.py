import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from osso.engine.engine import Engine

MODEL_PATH = "meta-llama/Llama-3.2-1B"
MAX_LENGTH = 1024
STRIDE = 512


@torch.inference_mode()
def evaluate_perplexity(
    engine: Engine,
    text: str,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
) -> float:
    enc = engine.tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(engine.device)
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_scored = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]

        logits = engine.model(window)
        shift_labels = window[:, 1:]
        n_to_score = min(trg_len, shift_labels.size(1))

        scored_logits = logits[0, -(n_to_score + 1):-1, :].float()
        scored_labels = shift_labels[0, -n_to_score:]
        loss = F.cross_entropy(scored_logits, scored_labels, reduction="sum")
        del logits, scored_logits

        nll_sum += loss.item()
        n_scored += n_to_score
        prev_end = end

        if end == seq_len:
            break

    return math.exp(nll_sum / n_scored)


def load_wikitext2_test() -> str:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def main():
    engine = Engine(MODEL_PATH)
    text = load_wikitext2_test()
    ppl = evaluate_perplexity(engine, text)
    print(f"Perplexity (WikiText-2 test): {ppl:.4f}")


if __name__ == "__main__":
    main()
