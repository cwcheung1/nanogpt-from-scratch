# 00 — Roadmap: the big picture before the weeds

Read this before lesson 1, and come back to it any time you feel lost in a
later lesson. It answers three questions the individual lessons assume
you've already got straight: **what are we actually building, why does it
take 6 steps, and what do all these recurring words mean** (embedding,
logits, loss, batch, block size, etc. — defined once here, used everywhere
else).

## What are we actually building?

**A tiny version of the same kind of model that powers ChatGPT/Claude.**
Not a toy that's conceptually different and simpler — the *actual*
architecture (a decoder-only Transformer), just scaled down enormously:
~11 million learned numbers instead of hundreds of billions, trained on
1MB of Shakespeare instead of a large fraction of the internet, running on
one gaming GPU for 13 minutes instead of a data center for months.

Small enough to fully understand, big enough that the same mechanisms
(attention, embeddings, residual connections, training loop) are exactly
what production models use. Nothing you learn here gets thrown away when
you eventually look at a real model's code — it's the same shapes, the same
few operations, just more of them.

**What "Transformer" actually means, mapped to the lessons you'll build**:
a *transformer* is a specific architecture (from the 2017 paper "Attention
Is All You Need") whose defining feature is **self-attention** (lesson 3)
as the mechanism positions use to exchange information — instead of older
architectures like RNNs, which process one position at a time in strict
order and can "forget" things over long distances. Concretely, a
transformer is: token + position embeddings (lesson 4) feeding into a
stack of **blocks** (lesson 5), each block being "communicate" (multi-head
self-attention, lessons 3-4) then "compute" (a per-position feedforward
MLP), wrapped in residual connections + LayerNorm, repeated `n_layer`
times, then a final output head. Lesson 6 is exactly this structure at
real scale. This project builds a **decoder-only** transformer specifically
(same family as GPT-2/GPT-4/Claude) — "decoder-only" means it only ever
looks backward at earlier positions (the causal mask, lesson 3), because
its one job is predicting what comes next. The original paper had an
encoder half too (built for translation); modern LLMs dropped it and kept
only the decoder side. Nothing about the training mechanism (loss,
`backward()`, the optimizer — see the glossary below) is transformer-
specific; only the architecture itself (attention + blocks) is.

## What does "language model" actually mean, mechanically?

Strip away every buzzword and a language model is a function with one job:

> Given some text so far, output a probability for every possible next
> character (or word/token, in bigger models) — i.e. "how likely is each
> possible thing that could come next?"

That's it. There's no understanding, no reasoning module, no separate
"generation" logic. **"Writing text" is just: ask the model for those
probabilities, randomly pick one next character weighted by those
probabilities, glue it onto the text so far, and repeat.** One character (or
token) at a time, forever. Every one of the 6 lessons is really just
building a better and better version of that one function — the thing that
looks at "text so far" and scores "what comes next."

The number we use to score "how good is this function" is called **loss**
(defined below) — lower is better, and it's the single scoreboard number
you'll see improve, lesson after lesson:

| after lesson | model | val loss | what it means |
|---|---|---|---|
| — | random guessing | ~4.17 | no learning at all — uniform over 65 characters |
| 2 | bigram (1 char of memory) | ~2.48 | learned *something* — letter frequencies, common pairs |
| 4 | attention (8 chars, first version) | ~2.33 | context beyond 1 character starts to help |
| 5 | transformer block, 4 layers | ~2.10 | stacking + residual connections pay off |
| 6 | full-size GPT (256 chars, 6 layers) | **1.48 (best)** | real scale, real regularization |

Every lesson exists because it fixes a specific, nameable limitation in the
previous one — not because "more complexity is generally good." Keep this
table in the back of your mind; it's the through-line connecting all 6.

## Glossary — the recurring words, defined once

These come up in almost every lesson. Rather than re-explain each one every
time, every lesson assumes you've read the definition here.

- **Tensor**: a multi-dimensional array of numbers (a 1D tensor is just a
  list; a 2D tensor is a grid/table; a 3D tensor is a stack of grids). It's
  PyTorch's basic data container — think "a Python list/array, but the
  library knows how to do fast, vectorized math on it, including on a GPU."
  A tensor's **shape** is just its dimensions, e.g. shape `(4, 8)` means "4
  rows, 8 columns."

- **Parameters / weights**: the actual numbers a model learns. Before
  training they're random; training's whole job is nudging them so the
  model's outputs get closer to correct. "An 11M-parameter model" means
  there are 11 million individual numbers being learned.

- **Neural network layer**: a function that takes a tensor in, does some
  fixed kind of math involving its own learnable weights, and produces a
  tensor out. `nn.Linear` (a learned matrix multiply) and `nn.Embedding` (a
  learned lookup table, below) are the two you'll see constantly. A model is
  just several layers chained together.

- **Embedding**: a lookup table that maps each item in a fixed vocabulary
  (e.g. each of the 65 possible characters) to a list of numbers (a
  "vector") that the model learns. Instead of the model seeing the number
  `12` (an arbitrary index), it sees a whole learned vector for "whatever
  character `12` represents" — giving the model room to encode *properties*
  of that character (which other characters it tends to appear near, etc.),
  not just its arbitrary ID.

- **Logits**: the model's raw, un-normalized output scores for "how likely
  is each possible next character" — one number per character in the
  vocabulary, before they've been turned into actual probabilities. Bigger
  logit = model thinks that character is more likely. They can be any
  positive or negative number; they don't sum to 1 yet.

- **Softmax**: the function that turns a row of logits into real
  probabilities — squashes every value into (0, 1) and makes the whole row
  sum to exactly 1. Applied whenever we need "an actual probability
  distribution to sample the next character from," not just relative
  scores.

- **Loss / cross-entropy loss**: one number summarizing "how wrong was the
  model's probability guess, on average, across every prediction in a
  batch." Cross-entropy specifically penalizes the model for putting low
  probability on the character that actually came next in the real text —
  lower loss = the model assigned higher probability to the right answer,
  on average. This is the single number training tries to shrink, and the
  number in the scoreboard table above.

- **Training loop**: the repeating cycle that actually improves the model:
  1. grab a batch of (history, correct-next-character) examples (lesson 1's
     `get_batch`)

  2. run them through the model to get logits, compute the loss
  3. **backward pass** (`loss.backward()`) — see the next entry, it's worth
     understanding, not skipping
  4. **optimizer step**: actually nudge every parameter a small step in the
     direction that reduces loss (`optimizer.step()`)
  4000+ repeats of this cycle is "training."

- **How `backward()` actually works** (took real back-and-forth to land the
  first time — read slowly). Think of one learnable number as a dial.
  Turning it changes a prediction, which changes how wrong you are (the
  loss). `backward()` answers exactly one question per dial: *"if I turned
  this up a tiny bit, would loss go up or down, and by how much?"* Proof,
  not assertion: `w=3, x=2, target=10` gives `pred=6`, `loss=16`, and
  `w.grad = -16`. Nudge `w` by `0.001` by hand (no autograd at all) and
  recompute the loss — you get almost exactly the same number back
  (`-15.996`). That's not a coincidence; it's the literal definition of a
  derivative, computed exactly, not approximated.

  This scales along 3 axes worth holding separately in mind:
  - **Many weights, not one**: every learnable number in the model (every
    entry in every embedding table and `Linear` weight matrix) gets this
    *exact same treatment*, completely independently — its own dial, its
    own gradient, computed simultaneously with every other one in a single
    `backward()` call.
  - **Many predictions, summed/averaged, not one**: real loss is an average
    over every prediction in a batch (`cross_entropy`, lesson 2). A weight
    shared across 2 predictions gets a gradient equal to the *average* of
    what each prediction alone would have pushed it toward — proven: one
    prediction alone gives gradient `-16`, another alone gives `70`, and
    the actual combined-loss gradient is exactly their average, `27`. A
    weight untouched by every prediction in a batch gets exactly `0` —
    nudging it couldn't have changed an output it was never used to
    produce.
  - **Many chained steps, not one**: a real weight sits several operations
    away from the loss, not right next to it. Tracing back through N steps
    is just multiplying together N simple local rules, one hop at a time
    (`effect of pred on loss` × `effect of h on pred` × `effect of w on h`).
    Proven across a 3-step chain (multiply → multiply → square): `backward()`'s
    answer, the hand-multiplied chain, and a finite-difference check across
    the *whole* chain all agree (`-64.0`, `-64.0`, `-63.98`).

  This whole "multiply local rules together, walking backward" process has
  a name: **backpropagation** (or reverse-mode automatic differentiation).
  It's the same mechanism underneath *every* neural network — nothing
  about it is specific to attention or transformers.

- **train / validation (val) split & overfitting**: `train_data` is what the
  loop above actually learns from. `val_data` is held out and never trained
  on — its only job is to be checked periodically as an honesty test: does
  the model do well on text it's never seen, or did it just memorize the
  training text? If train loss keeps falling but val loss stops falling (or
  rises), that gap is called **overfitting** — the model is memorizing
  specifics instead of learning generalizable patterns. You'll see this
  happen for real in lesson 6.

- **`block_size`** (context length): how many characters of history the
  model is shown before predicting the next one. A hard limit, not a
  preference — the model is physically incapable of using more context than
  this, by construction.

- **`batch_size`**: how many independent, unrelated training examples get
  bundled together and processed as one unit, purely for hardware speed
  (a GPU multiplies a batch of N examples through the network in about the
  same time as 1). Doesn't change what's learned, only training speed/
  stability. See lesson 1 for the full "parallel means parallel on the
  hardware, not related to each other" breakdown.

- **`n_embd`** (the embedding dimension — also called `d_model` in the
  original Transformer paper, or `hidden_size` in Hugging Face configs):
  how many numbers describe *each single character*. Easy to confuse with
  `vocab_size` (65) since both are "just a number near the code that talks
  about characters" — they are **not** the same axis. `vocab_size` is how
  many *distinct* characters exist (fixed by the data); `n_embd` is how
  richly *each one* gets described (32 in lessons 4-5, 384 in lesson 6 —
  free to change, unrelated to `vocab_size`). A shape like `(1, 8, 32)` is
  1 sequence, 8 characters, 32 numbers *per* character — the `32` is not
  "32 more characters."

- **Is this number a choice, or fixed, or computed?** Every hyperparameter
  you meet falls into exactly one of three buckets — worth asking every
  time one shows up:
  1. **Data-fixed** — determined by the dataset, not a choice (`vocab_size`).
  2. **Arbitrary design choice** — picked by whoever wrote the code, no
     formula behind it (`n_embd`; lesson 3's standalone `head_size = 16`).
  3. **Formula-derived** — computed from other choices (lesson 4+'s
     `head_size = n_embd // n_head` — the real free choice is `n_head`,
     `head_size` just falls out of the division).
  Also worth internalizing early: none of the numbers *inside* a learned
  vector (an embedding, or a query/key/value vector) has an assigned
  meaning — nobody decides "dimension 7 means X." Every one starts as
  noise and becomes whatever training finds useful. A width like `n_embd`
  or `head_size` controls *how many* such free numbers a component gets
  to work with, not *which specific things* get computed in which slot.

## PyTorch/Python idioms — the code-level words, not the ML-concept words

The glossary above defines *machine-learning* terms (logit, loss, embedding).
This section defines the *PyTorch/Python* syntax that shows up in literally
every lesson's code, regardless of which ML concept that lesson teaches.
These aren't concepts to sit and think about — they're small, fixed pieces
of syntax, defined once so no lesson has to stop and re-explain them.

- **`torch`**: the library itself. Provides tensors (see glossary above)
  and fast, GPU-capable math on them — think "numpy, but tensors can live on
  a GPU and PyTorch can automatically compute derivatives for them."

- **`torch.nn`** (imported as `nn`): the part of `torch` that provides
  ready-made *layers* — reusable pieces of math with their own learnable
  numbers already wired up (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
  ...). You assemble a model out of these instead of writing raw matrix
  multiplication by hand every time.

- **`nn.Module`**: the base class every model/layer in this repo inherits
  from (`class BigramLanguageModel(nn.Module)`). Inheriting from it is what
  lets PyTorch automatically discover "which numbers inside this object are
  learnable parameters" so the optimizer can find and update them — you
  don't do that bookkeeping by hand.

- **`super().__init__()`**: plain Python, not PyTorch-specific — every
  `__init__` that inherits from a parent class calls this first, to run the
  parent class's (here, `nn.Module`'s) own setup before adding your own
  layers on top. Always the first line inside `__init__` in this repo.

- **`optimizer` (`torch.optim.AdamW`)**: the algorithm that turns "which
  direction would reduce the loss" (computed by `loss.backward()`, see
  Training loop above) into an actual change to every parameter. AdamW is
  just the specific, standard variant used here and in GPT-2/nanoGPT —
  treat its internals as a black box; every lesson's training loop is
  `zero_grad()` → `backward()` → `step()`, always in that order.

- **`.to(device)`**: moves a tensor or model's numbers into GPU memory (if
  `device == "cuda"`) so the GPU does the math instead of the CPU. Every
  lesson computes `device = "cuda" if torch.cuda.is_available() else
  "cpu"` once near the top and calls `.to(device)` on the model and on each
  batch of data.

- **`dim=-1`**: "the last dimension of this tensor." For a `(B, T, C)`
  logits tensor, `dim=-1` means "operate across the `C` values" — e.g.
  `F.softmax(logits, dim=-1)` turns each position's `C` raw logits into `C`
  probabilities that sum to 1, independently at every `(B, T)` position.

- **`torch.manual_seed(1337)`**: fixes every "random" number PyTorch
  generates from that point on to a specific, reproducible sequence. Without
  it, two runs of identical code would get different random batches/weight
  initializations and produce different loss numbers — you'd have no way to
  tell if a change you made actually helped, or you just got a luckier
  random draw. Every lesson's script calls this near the top for exactly
  this reason.

- **`.tolist()` / `.item()`**: convert a tensor back into plain Python
  (`.tolist()` → a Python list, for feeding into `decode()`; `.item()` → a
  single Python number, for something like `loss.item()`). Tensors have
  their own numeric type; these calls step back out into ordinary Python
  values for printing or further plain-Python use.

- **`@torch.no_grad()`** (seen from lesson 4 on, decorating `generate`):
  tells PyTorch "don't bother tracking gradients for anything in this
  function" — gradient tracking is only needed during training
  (`loss.backward()`), and skipping it during text generation is faster and
  uses less memory.

## The 6-lesson journey — what each one adds, and why the previous one wasn't enough

1. **[Tokenization & data prep](01-tokenization-and-data.md)** — no model
   yet. Just: how does text become numbers (encode/decode), and how do we
   manufacture the (history → correct-next-character) flashcards every
   later lesson trains on (`get_batch`). Everything downstream depends on
   this data format.

2. **[Bigram baseline](02-bigram-baseline.md)** — the simplest possible
   model: one lookup table, predicting the next character using *only* the
   single character right before it (zero memory beyond that). Val loss
   ~2.48. **The problem this exposes**: it has a hard ceiling, because it's
   architecturally blind to anything more than 1 character back — no amount
   of extra training fixes that, only a different architecture can.

3. **[Self-attention mechanics](03-self-attention.md)** — introduces the
   mechanism that lets a position pull information from *every* earlier
   position, weighted by *learned* relevance, instead of being stuck at 1.
   Not trained on real data yet — this lesson is purely "build the
   mechanism and inspect it," so you can see the actual attention weights
   before worrying about training dynamics on top. **The problem this
   exposes**: one attention head only extracts one kind of pattern; also,
   attention alone has no sense of word/character *order*.

4. **[Multi-head attention](04-multi-head-attention.md)** — runs several
   attention heads in parallel (different heads can specialize in different
   kinds of relationships) and adds position embeddings (fixing the
   "no sense of order" gap). First lesson where attention is actually
   trained on real data. Val loss improves to ~2.33 — a real, measured win
   over the bigram's ~2.48 ceiling, from using context beyond 1 character.
   **The problem this exposes**: attention alone doesn't let you *stack*
   many layers deep without training becoming unstable.

5. **[The transformer block](05-transformer-block.md)** — wraps attention
   with a per-position "thinking" MLP, residual (skip) connections, and
   LayerNorm, then stacks 4 of these blocks. Val loss improves to ~2.10.
   Includes a deliberate side-by-side experiment training the *same*
   architecture with and without residual connections, so you watch (not
   just read) why they matter — without them, loss plateaus worse than the
   bigram baseline despite 13x more parameters. **The problem this
   exposes**: this is a small, undertrained proof of concept (32-dim
   embeddings, 8-character context) — it works, but it's not yet at a scale
   or with the training details (dropout, gradient clipping) a real run
   needs.

6. **[Full GPT, training, sampling](06-full-gpt-and-training.md)** — same
   architecture as lesson 5, scaled up to nanoGPT-tutorial size (384-dim
   embeddings, 256-character context, 6 layers, 6 heads, ~11M params) with
   the regularization/stability details a real run needs (dropout, gradient
   clipping, GPT-2-style weight init), trained for 5000 steps. Val loss
   bottoms out at **1.48**, then — left running — visibly overfits, which
   you watch happen in the numbers rather than take on faith. This is the
   finish line for *pretraining mechanics*; see below for what's
   deliberately not covered.

## What this project deliberately does NOT cover

So you don't wonder if you missed it:

- **BPE (subword tokenization)** — real models don't use one-token-per-
  character; see lesson 1's writeup for what BPE is and why this project
  skips it on purpose.

- **Fine-tuning / instruction-tuning / RLHF** — this project only covers
  *pretraining* (learning raw next-token prediction from a big pile of
  text). Turning that into something that follows instructions and chats
  helpfully is a separate, later stage — a different repo, not started yet.

- **Serving/inference optimization** (KV-caching, quantization, batching
  requests from multiple users) — mentioned in passing (lesson 2 notes what
  the bigram model's wastefulness foreshadows), not built here.

## How to use this roadmap day-to-day

- Stuck on a specific term mid-lesson? Check the glossary above first —
  most jargon is defined once, here, not re-derived in every lesson.

- Confused about *why* a lesson exists at all? Re-read its entry in "The
  6-lesson journey" above — every lesson is motivated by a specific,
  nameable gap in the previous one.

- Lost in the code? Every lesson's script is heavily commented line-by-line
  — read the comments as a narration of what's happening and why, not just
  a label.

- The scoreboard table near the top is the thing to hold onto when
  everything else feels like a lot of moving parts: **one number, getting
  better, one idea at a time.**
