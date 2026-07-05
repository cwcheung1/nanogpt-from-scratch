# nanoGPT from scratch — lesson index

Learning path: build a GPT-style language model from nothing, one concept at
a time, in the style of Andrej Karpathy's nanoGPT / "Let's build GPT" —
each lesson is a runnable script plus a writeup that ties the code to the
underlying concept. Work through in order; each stage's code and results are
compared directly against the previous stage's.

1. [Tokenization & data prep](01-tokenization-and-data.md) — char-level
   tokenizer, train/val split, the `get_batch()` function every later lesson
   reuses.
2. [Bigram baseline model](02-bigram-baseline.md) — simplest possible neural
   LM (one embedding table, no context beyond 1 token). The floor everything
   else has to beat. Val loss plateau: ~2.48-2.50.
3. [Self-attention mechanics](03-self-attention.md) — the averaging "trick"
   built up three equivalent ways, then a real Q/K/V self-attention head
   with causal masking, inspected directly (not yet trained).
4. [Multi-head attention](04-multi-head-attention.md) — several attention
   heads in parallel + position embeddings, wired into a real trainable
   model for the first time. Val loss: ~2.33-2.36.
5. [The transformer block](05-transformer-block.md) — attention + MLP +
   residual connections + pre-LayerNorm, stacked 4 deep. Includes an
   ablation training the same architecture *without* residuals side by side,
   to see empirically why they matter. Val loss: ~2.10 (with residuals) vs.
   ~3.3, noisy (without).
6. [Full GPT, training loop, sampling](06-full-gpt-and-training.md) — scaled
   up to nanoGPT-tutorial size (10.8M params, 6 layers, 6 heads, 384 embd,
   256 context), dropout, gradient clipping, GPT-2-style init, full training
   run, checkpoint saved, and real sampled text.

Every lesson's code lives in `lessons/code/`, imports shared data-loading
utilities from `lessons/code/common.py`, and is runnable via `make lessonN`.
See the repo's `NOTES.md` for the running progress/status log and
`README.md` for setup.
