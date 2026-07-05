# 02 — Bigram Baseline Model

**Concept**: the simplest possible neural language model predicts the next
character using *only* the current character, via one learned lookup table
— no attention, no hidden layers, no context beyond a single token.

**Analogy**: imagine a giant 65x65 table where row *i* is "if the current
character is `i`, here's how likely every other character is to come next."
That table, learned from data, is the entire model. It's the "always guess
the most common follow-up letter" strategy, made probabilistic and trainable.

**How it works** (`lessons/code/02_bigram_baseline.py`):
- `nn.Embedding(vocab_size, vocab_size)` — normally an embedding table maps
  token id -> a dense vector. Here the output dimension is *also*
  `vocab_size`, so each row directly **is** the logits over "what comes
  next." There's no separate output projection because the embedding table
  doubles as one.
- `forward` reshapes `(B, T, C)` -> `(B*T, C)` before `F.cross_entropy`,
  because PyTorch's cross entropy wants one prediction-vs-target pair per
  row, and we have `B*T` independent next-character predictions per batch
  (this is the "8 training signals per row" idea from lesson 1, now actually
  consumed by a loss function).
- `generate` calls the model repeatedly, taking `softmax` over the logits
  for the *last* position and sampling one character via
  `torch.multinomial` (not `argmax` — sampling keeps output varied instead
  of deterministically repeating the single most likely character forever).
  Note the comment in the code: this model recomputes the whole sequence
  every step but only ever looks at the last token anyway, because a bigram
  model has no mechanism to use earlier context even if you gave it some.
  That wastefulness becomes relevant later — it's exactly what KV-caching in
  real inference serving fixes, once the model actually has multi-token
  context worth reusing.
- `estimate_loss` (in `common.py`) averages loss over 200 random batches
  instead of trusting one batch's loss, which is noisy. Watch both `train`
  and `val` loss printed side by side — this is your first look at
  over/underfitting diagnostics.

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
