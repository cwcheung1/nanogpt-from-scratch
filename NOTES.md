# NOTES — nanoGPT from scratch

Running progress/status log. Teaching material lives in `lessons/*.md`; this
is the denser stage-tracking reference (same convention as `adk-agent`).

## Status: all 6 lessons built and verified (2026-07-05)

Goal: learn the mechanics of training an LLM from scratch — architecture,
attention, training loop — on real frameworks (PyTorch), following Karpathy's
nanoGPT/"Let's build GPT" progression, verified by actually running each
stage rather than trusting the explanation.

Hardware: RTX 4070 SUPER (12GB VRAM), CUDA confirmed working (`torch==2.12.1+cu130`).

## Stage checklist

- [x] **Repo scaffold** — standalone git repo, `uv` project, tiny-shakespeare
      dataset (`data/input.txt`, 1.1M chars, 65-char vocab), Makefile.
- [x] **Lesson 1 — tokenization & data**: char tokenizer, train/val split,
      `get_batch`. Verified: encode/decode round trip, batch shapes, unrolled
      context->target example.
- [x] **Lesson 2 — bigram baseline**: single embedding table = whole model.
      Verified: val loss plateaus at **~2.48-2.50** after ~1500 steps.
- [x] **Lesson 3 — self-attention mechanics**: the "averaging trick" proven
      equivalent 3 ways (loop / matmul / masked-softmax), then a real Q/K/V
      `Head` with causal masking. Verified: `torch.allclose` across all 3
      trick variants, attention matrix confirmed lower-triangular with rows
      summing to 1.
- [x] **Lesson 4 — multi-head attention**: parallel heads + position
      embeddings, first trainable attention model. Verified: val loss
      **~2.33-2.36**, a real improvement over lesson 2's bigram floor.
- [x] **Lesson 5 — transformer block**: attention + MLP + residual + pre-LN,
      stacked 4 deep. Verified empirically via ablation — **with** residuals,
      val loss reaches **~2.10**; **without** residuals (same architecture,
      same seed, only the skip connection removed), val loss plateaus worse
      at **~3.2-3.3** and is noisy/non-monotonic late in training. This is
      the strongest single result in the series for *why* residuals matter.
- [x] **Lesson 6 — full GPT + training + sampling**: scaled to
      nanoGPT-tutorial config (10.8M params, n_layer=6, n_head=6, n_embd=384,
      block_size=256, dropout=0.2, grad clipping, GPT-2-style init), full
      5000-iteration training run (752s on the RTX 4070 SUPER), checkpoint
      saved to `checkpoints/nanogpt_shakespeare.pt` (gitignored). **Best val
      loss 1.48 at step ~3000; run continued past that to step 5000 and
      visibly overfit** (train loss kept falling to 0.84, val loss rose back
      to 1.58) — caught empirically, not just described. Sampled text is
      clearly Shakespeare-*shaped* (correct character-name/dialogue
      formatting, verse-like structure) though not coherent at the
      paragraph level. See `lessons/06-full-gpt-and-training.md` for the
      full step-by-step loss table.

## Open / next steps (not part of this repo — see roadmap memory)

This repo covers **pretraining mechanics only**. Deliberately out of scope
here, planned as the next leg of the learning path:
1. **Fine-tuning** a real open-weight model (Llama 3 8B or Qwen2.5 7B) with
   QLoRA (`transformers` + `peft` + `bitsandbytes`) — separate repo.
2. **Serving** — vLLM or llama.cpp, KV caching, batching, quantization —
   separate repo.
3. Gaps nanoGPT alone doesn't cover: alignment (RLHF/DPO/instruction
   tuning), tokenizer training itself (BPE from scratch — Karpathy's
   `minbpe` is the reference for this if wanted later).
