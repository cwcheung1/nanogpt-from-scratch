# 06 — Full GPT, Training Loop, and Sampling

**Concept**: assemble everything from lessons 1-5 into the actual
nanoGPT-tutorial configuration (10.8M params: 6 layers, 6 heads, 384-dim
embeddings, 256-token context), add the regularization/stability details a
real training run needs (dropout, gradient clipping, GPT-2-style init), and
run it long enough to see both a good result *and* a real failure mode
(overfitting) show up in the loss curve.

**Analogy**: lessons 2-5 were building and testing a car's components one
at a time in a garage — engine, transmission, brakes, all bench-tested
individually. This lesson is putting them all together in the actual car
and driving it far enough to learn something only a real drive teaches you:
here, that more training isn't free — past a point, the model starts
memorizing the training set's specific sentences instead of learning
general structure, and gets *worse* at everything else.

**How it works** (`lessons/code/06_full_gpt_and_training.py`):
- Architecture is identical to lesson 5's `Block`/`GPTLanguageModel` — same
  attention, same MLP, same residual+pre-LN pattern — just scaled up
  (`n_embd` 32→384, `n_head` 4→6, `n_layer` 4→6, `block_size` 8→256).
  Nothing conceptually new in the architecture itself; this lesson is about
  what changes when you scale it and train it properly.
- **Dropout** (`nn.Dropout(0.2)`, added inside `Head`, `MultiHeadAttention`,
  and `FeedForward`): randomly zeroes some values during training so the
  model can't over-rely on any single attention weight or MLP unit being
  present — a direct regularizer against the overfitting this lesson's run
  demonstrates.
- **Gradient clipping** (`torch.nn.utils.clip_grad_norm_(model.parameters(),
  1.0)`): caps the total gradient norm before the optimizer step, preventing
  any single unusually large batch from producing a destabilizing update —
  standard practice at this scale and up.
- **GPT-2-style init** (`_init_weights`): small (`std=0.02`) normal
  initialization for `Linear`/`Embedding` weights, zero bias. Not
  load-bearing for a model this small, but it's the actual scheme GPT-2 and
  nanoGPT use, worth having muscle memory for.
- Checkpoint saved via `torch.save(model.state_dict(), ...)` — the standard
  PyTorch way to persist trained weights (gitignored here since it's a
  binary artifact, not source).

**Look at**: `lessons/code/06_full_gpt_and_training.py`

**Run this**:
```
make lesson6      # takes ~12-13 minutes on an RTX 4070 SUPER
```

Actual run (5000 iterations, 752s total on this machine):

| step | train loss | val loss |
|---|---|---|
| 0 | 4.22 | 4.23 |
| 500 | 1.70 | 1.86 |
| 1000 | 1.37 | 1.60 |
| 2000 | 1.18 | 1.50 |
| **3000** | 1.06 | **1.48 (best)** |
| 4000 | 0.95 | 1.51 |
| 4999 | 0.84 | 1.58 |

**This is overfitting, caught in the act, not asserted**: train loss falls
monotonically the entire run, but val loss bottoms out around step 3000 and
then *rises* while train loss keeps dropping — the model is increasingly
memorizing specific sequences from the 1M-character training set rather
than learning generalizable structure, and the growing train/val gap is the
textbook signature. (The fix — early stopping at the val-loss minimum,
more dropout, more data, or a smaller model for this dataset size — is a
direct application of what dropout is already doing here, just needing more
of it or less training past step ~3000.)

The sampled text past that overfitting point is still notably better
structured than any earlier lesson: character names in the right format
(`LUCENTIO:`, `EDWARD:`), plausible verse-like line breaks, dialogue that's
locally grammatical even where it's globally nonsensical. That's the
ceiling of a 10.8M-parameter *character-level* model trained on 1MB of
text — genuinely instructive as a contrast against what scale (GPT-2's
124M-1.5B params, web-scale data, subword tokens) buys you.

**You'll know it clicked when**: you can point to the exact step range
where this run crosses from "still learning" to "overfitting" using only
the train/val loss numbers (not the sampled text), and explain what
specifically about dropout (`p=0.2` here) would need to change to push that
crossover point later — and why "just train longer" is not a fix once val
loss is already rising.
