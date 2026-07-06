# nanoGPT from scratch

Learning project: build a GPT-style language model from nothing, following
Andrej Karpathy's nanoGPT / "Let's build GPT" progression — bigram baseline
→ self-attention → multi-head attention → transformer blocks → full GPT —
to learn the actual mechanics of LLM pretraining (not just call an API for
one) using the same core frameworks (PyTorch) real labs use.

This is the first leg of a two-part learning path: **pretrain from scratch**
(this repo) → **fine-tune + serve a real open-weight model** (separate repo,
not started yet).

## Setup

```
make setup      # uv sync
```

Requires an NVIDIA GPU for reasonable training speed (verified against an
RTX 4070 SUPER, 12GB VRAM). Runs on CPU too, just much slower.

## Working through it

Start at [`lessons/00-roadmap.md`](lessons/00-roadmap.md) — the big picture
before the weeds: what we're actually building, every recurring term
(embedding, logits, loss, batch/block size, etc.) defined once, and why each
of the 6 lessons exists. Then [`lessons/README.md`](lessons/README.md)
indexes all 6 lessons in order. Each lesson is a `lessons/NN-concept-name.md`
writeup plus a heavily-commented, runnable script in `lessons/code/`:

```
make lesson1    # tokenization & data prep
make lesson2    # bigram baseline
make lesson3    # self-attention mechanics
make lesson4    # multi-head attention
make lesson5    # transformer block (+ residual-connection ablation)
make lesson6    # full GPT: scaled up, trained, sampled
```

`NOTES.md` has the running progress/status log with actual verified
loss numbers from each stage.
