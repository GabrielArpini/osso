import torch
import torch.nn.functional as F
from osso.config import SamplingParams
from osso.kvcache.naive import NaiveKVCache


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    score = logits.gather(-1, generated)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(-1, generated, score)


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.float() / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < values[..., -1, None], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = (cumprobs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(engine, prompt: str, params: SamplingParams | None = None) -> str:
    if params is None:
        params = SamplingParams()

    input_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(engine.device)
    generated = input_ids
    needed_seq_len = input_ids.shape[1] + params.max_new_tokens
    kv_cache = NaiveKVCache(engine.config, engine.device, engine.dtype, max_seq_len=needed_seq_len)

    # prefill: process entire prompt, cache all K/V
    logits = engine.model(input_ids, kv_cache)
    kv_cache.advance(input_ids.shape[1])
    next_logits = logits[:, -1, :]
    if params.repetition_penalty != 1.0:
        next_logits = apply_repetition_penalty(next_logits, generated, params.repetition_penalty)
    next_token = sample(next_logits, params.temperature, params.top_k, params.top_p)
    generated = torch.cat([generated, next_token], dim=1)

    if next_token.item() == engine.tokenizer.eos_token_id:
        return engine.tokenizer.decode(generated[0], skip_special_tokens=True)

    # decode: one token at a time, reusing cached K/V
    for _ in range(params.max_new_tokens - 1):
        logits = engine.model(next_token, kv_cache)
        kv_cache.advance(1)
        next_logits = logits[:, -1, :]
        if params.repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, generated, params.repetition_penalty)
        next_token = sample(next_logits, params.temperature, params.top_k, params.top_p)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == engine.tokenizer.eos_token_id:
            break

    return engine.tokenizer.decode(generated[0], skip_special_tokens=True)
