# 02 — Bigram Baseline Model

*Before this lesson: read [00 — Roadmap](00-roadmap.md) if you haven't —
this lesson assumes you already know what "loss", "embedding", "logits",
and "the training loop" mean, and won't re-explain them.*

**Jargon buster — new terms this lesson's code uses** (full definitions in
the roadmap's [PyTorch/Python idioms](00-roadmap.md#pytorchpython-idioms--the-code-level-words-not-the-ml-concept-words)
section; this is just a preview so nothing below stops you cold):

- `nn.Module` — the base class `BigramLanguageModel` inherits from; lets
  PyTorch find this model's learnable numbers automatically.

- `super().__init__()` — plain Python: run the parent class's own setup
  first, before adding this model's own layer.

- `optimizer` / `AdamW` — the algorithm that turns "which direction reduces
  loss" into an actual update to the model's numbers. Black box for now.

- `.to(device)` — moves the model/data onto the GPU, if one's available.
- `dim=-1` (in `F.softmax(logits, dim=-1)`) — "operate across the last
  dimension" — here, the 65 per-position logits.

- `torch.manual_seed(1337)` — makes this script's "random" numbers
  reproducible run-to-run.

- `.tolist()` — converts a tensor back into a plain Python list, so
  `decode()` can use it.

**Concept**: the first actual model in this repo (lesson 1 had none). The
simplest possible neural language model predicts the next character using
*only* the single character right before it — no attention, no hidden
layers, no memory of anything further back. This is the floor: every later
lesson's whole point is beating this number.

**Analogy**: imagine a giant 65×65 table where row *i* is "if the current
character is `i`, here's how likely every other character is to come next."
That table, learned from data, is the *entire* model — there's nothing else
to it. It's the "always guess the most common follow-up letter" strategy,
made probabilistic and trainable.

**How it works** (`lessons/code/02_bigram_baseline.py`, now heavily
commented — read the comments alongside this):

- `nn.Embedding(vocab_size, vocab_size)` — an embedding (see roadmap
  glossary) normally maps a token id to a learned vector of some other size.
  Here the output size is *also* `vocab_size`, so each row of the table
  directly **is** the logits for "what comes next" — there's no separate
  output layer, because this one lookup table doubles as the whole model.

- `forward` reshapes the `(B, T, C)` logits tensor into `(B*T, C)` before
  computing the loss, because PyTorch's `F.cross_entropy` wants one
  prediction-vs-answer pair per row, and we have `B*T` independent
  next-character predictions in one batch (this is lesson 1's "one 8-char
  window secretly gives 8 flashcards" idea, now actually being scored by a
  loss function for the first time).

- `generate` calls the model repeatedly: get logits for the last position →
  `softmax` turns them into real probabilities → `torch.multinomial` picks
  one character *randomly, weighted by those probabilities* (not the single
  most-likely character every time — that would repeat forever; sampling
  keeps output varied) → glue it onto the sequence → repeat. This is
  literally the "ask the model, sample, repeat" loop from the roadmap's
  "what is a language model" section, running for real.
  One inefficiency worth clocking now because it becomes relevant later:
  `generate` re-runs the *entire* sequence through the model at every single
  step, even though a bigram model only ever looks at the last character
  regardless of how much history you hand it. That wasted recomputation is
  exactly what KV-caching (a real inference-serving optimization, not
  covered in this repo) exists to eliminate — once a model actually has
  multi-character context worth reusing, which this one doesn't yet.

- `estimate_loss` (in `common.py`) averages the loss over 200 random
  batches instead of trusting a single batch's loss, which jumps around too
  much to read reliably. Watch `train` and `val` loss printed side by side —
  this is your first live look at the over/underfitting check from the
  roadmap glossary.

**Look at**: `lessons/code/02_bigram_baseline.py`, `estimate_loss` in
`lessons/code/common.py`

**Run this**:
```
make lesson2
```
Expect: loss starts around `ln(65) ≈ 4.17` (uniform-random guessing over 65
characters) and drops to a plateau around `2.4-2.5` after ~1500 steps — it
stops improving because a single-character lookup table has a hard
information ceiling. Output goes from pure noise to vaguely
word-shaped-but-meaningless text (correct letter *frequencies* and common
digrams, zero sense of longer structure).

**You'll know it clicked when**: you can explain *why* this model's loss
plateaus no matter how long you train it (hint: what information is
physically available to `self.token_embedding_table(idx)` at each position?)
— and why fixing that requires the model to see more than one token of
context, which is exactly what lesson 3's self-attention adds.
