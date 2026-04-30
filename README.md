# osso

A general-purpose LLM inference engine with online and offline serving, built for low-end consumer GPUs.

Most inference engines are designed around datacenter hardware — 80GB VRAM, multi-GPU clusters, high-end Ampere and Hopper architectures. osso takes the opposite approach: a single consumer GPU with tight memory constraints, where every byte of KV cache and every memory access pattern actually matters. The kind of hardware most people actually have.

The engine targets Turing (SM75 / RTX 20xx) as its primary architecture, a generation that existing optimized kernels like FlashAttention largely ignore. Everything from the attention kernel to the KV cache is designed around the realities of running on 6GB of VRAM,not as a limitation, but as the actual design constraint.

## References

- **Llama 3**: Llama Team, AI @ Meta. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024.
- **Llama 2**: Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.
- **RMSNorm**: Zhang & Sennrich. "Root Mean Square Layer Normalization." arXiv:1910.07467, 2019.
- **RoPE**: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864, 2021.
