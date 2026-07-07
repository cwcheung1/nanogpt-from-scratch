# 03 — Self-Attention Mechanics

*Before this lesson: [00 — Roadmap](00-roadmap.md) and lesson 2. This is
the densest lesson in the series conceptually — expect to reread it. There
is no trained model in this lesson; we're only building and inspecting the
mechanism, on random data, so training details can't distract from what
attention itself actually computes.*

**Jargon buster — new terms this lesson's code uses** (full definitions in
the roadmap's [PyTorch/Python idioms](00-roadmap.md#pytorchpython-idioms--the-code-level-words-not-the-ml-concept-words)
section; `nn.Module`/`super().__init__()`/`dim=-1` were already covered in
lesson 2's jargon buster and won't be repeated here):

- **`nn.Linear(in, out, bias=False)`** — a learned linear transformation
  (`y = x @ W`, no `+ b` since `bias=False`); `key`/`query`/`value` are each
  one of these.

- **`register_buffer`** — stores a tensor as part of the model (so `.to(device)`
  moves it to the GPU along with everything else) without treating it as a
  learnable parameter — used here for the fixed causal mask, which never
  changes during training.

- **`.transpose(-2, -1)`** — swaps a tensor's last two dimensions (needed to
  line up `q` and `k` for the matrix multiply below).

- **`@`** — Python's matrix-multiplication operator; `q @ k.transpose(-2,-1)`
  computes every query's dot product against every key at once.

- **`.masked_fill(cond, value)`** — replaces every element where `cond` is
  true with `value` (here: replace future positions with `-inf` before
  softmax, so they become exactly 0 probability).

- **`torch.allclose(a, b)`** — checks two tensors are equal within floating-
  point rounding error (exact `==` is unreliable for computed floats).

**Concept, in one sentence**: for every position in a sequence, look at
every *earlier* position, decide how much each one matters *right now*
(a "relevance score" the model learns), and blend them together weighted
by those scores. That's it — that whole sentence is what "self-attention"
means. It's a strict upgrade over lesson 2's bigram, which could only ever
look at exactly 1 character back with a fixed, non-learned rule.

**The mechanical version, first — no analogy yet**: each position produces
3 vectors (via 3 separate learned linear layers — see roadmap glossary on
`nn.Linear`) called query, key, and value. "Relevance score between
position A and position B" = a dot product of A's query vector with B's key
vector (dot product is just "how aligned are two vectors" — two vectors
pointing the same direction score high, two pointing in unrelated
directions score near 0). Turn all those raw scores into real probabilities
with softmax (every row now sums to exactly 1), then use those
probabilities as weights to take a weighted average of everyone's value
vectors. Query/key decide the *weights* (how much to listen to each
position); value is *what gets averaged*. That's the entire mechanism —
everything else below is either terminology for a piece of it, or code that
implements it.

One term that comes up a lot below: a **lower-triangular matrix** is just a
square grid of numbers where everything *above* the diagonal (top-left to
bottom-right) is zero, and everything on/below it can be non-zero. Visually,
for a 4×4 example (`1`s below/on the diagonal, `0`s above):
```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```
Row *i* has non-zero entries only in columns `0..i` — which is exactly
"positions I'm allowed to look at: myself and everything before me, nothing
after." That shape is why this matrix keeps showing up: it's a compact way
to say "no cheating by looking at the future" in matrix form.

**If an analogy helps**: picture a group discussion where, before speaking,
each person silently asks "who here said something relevant to what I
need?" (their query), everyone else holds up a sign summarizing what they
said (their key), the person compares their question against every sign to
get relevance scores (this is the dot-product step above), and gathers a
weighted blend of what those people *actually meant to share* (their
value) — not the sign, the real message. Query/key decide *how much* to
listen to whom; value is *what* gets passed along.

**Why `* head_size**-0.5` — shown, not asserted.** The code scales every
raw dot product down by `head_size**-0.5` before softmax. Here's what
actually happens without that scaling, using real numbers from random
query/key vectors (same random seed, only `head_size` changes):

```
head_size=  16: raw dot product=    2.98   scaled=  0.75
head_size=  64: raw dot product=   -4.72   scaled= -0.59
head_size= 256: raw dot product=  -14.58   scaled= -0.91
```

Raw dot products get bigger in magnitude as `head_size` grows — each of the
`head_size` dimensions contributes its own bit of randomness to the sum, so
the total swings further from 0 the more dimensions you add. `* head_size
**-0.5` cancels that growth back out, regardless of `head_size`.

Why this matters for softmax, concretely — same 5 mildly-different scores,
first fed to softmax as-is, then artificially blown up the way an
unscaled, large-`head_size` dot product would be:

```
head_size= 256 (unscaled): softmax = [0.032, 0.774, 0.006, 0.156, 0.032]
with scaling applied:      softmax = [0.191, 0.233, 0.173, 0.211, 0.191]
```

Same underlying preferences, wildly different softmax output. Unscaled,
one position grabs 77% of the attention weight and three others get
crushed under 4% — the model would end up attending to almost one
position only, even though the raw scores barely favored it. Scaled, the
distribution stays close to the actual relative preferences. This is
"softmax saturation," and it's why every real transformer (GPT-2, GPT-4,
Claude's public architecture descriptions) scales by `1/sqrt(head_size)` —
without it, larger head sizes would make attention increasingly all-or-
nothing regardless of how the model actually wants to weigh its options.

**How it works** (`lessons/code/03_self_attention.py`):

- **Part A — "the trick"**: three functions that compute the *identical*
  result three ways, verified with `torch.allclose`:

  - `trick_v1_loop` — for each position, literally average the embeddings of
    every position up to and including it. Correct, but a slow Python loop.

  - `trick_v2_matmul` — the same averaging, but as one matrix multiply: a
    lower-triangular matrix of 1s, each row normalized to sum to 1, *is* the
    "average everyone before me" operator. `wei @ x` replaces the double
    loop entirely.

  - `trick_v3_softmax` — same result again, but built from `masked_fill(...,
    -inf)` followed by `softmax`. This form looks like more work for the
    same answer, but it's the one that generalizes: replace the all-zeros
    `wei` with *learned, data-dependent* scores, and softmax turns arbitrary
    scores into a valid weighted average. The `-inf` mask is what forces
    future positions to get exactly zero weight after softmax — this is
    "causal" masking, the mechanism that prevents a language model from
    cheating by looking at the answer.

- **Part B — a real `Head`**: `key`, `query`, `value` are three separate
  learned linear projections of the same input `x` (no bias — matches
  nanoGPT/GPT-2). `wei = q @ k.transpose(-2,-1) * head_size**-0.5` is the
  scaled dot-product: dot product measures alignment between what position
  *i* is looking for and what position *j* offers; the `head_size**-0.5`
  scaling keeps the dot products from growing large enough to make softmax
  saturate (push everything to 0/1) as `head_size` grows — this is the
  literal "scaled" in "scaled dot-product attention." Then: causal mask,
  softmax (each row is now a real probability distribution, not a hardcoded
  average), then `wei @ v` gathers the weighted blend of *values* (not keys —
  that distinction matters, keys are for matching, values are what actually
  gets passed through).

- The printed weight matrix for batch element 0 is the thing to actually
  read: row 0 must be `[1, 0, 0, ...]` (only itself exists yet), row 1 splits
  between positions 0-1, and so on — but the *split* is no longer 50/50 like
  `trick_v2`, it's whatever the learned query/key vectors produced (0.40/0.60
  in the run above, before any training has happened).

**Look at**: `lessons/code/03_self_attention.py`

**Run this**:
```
make lesson3
```
Expect: `v1 vs v2 match: True`, `v2 vs v3 match: True`, a `Head` output shape
of `(4, 8, 16)`, an attention weight matrix that's lower-triangular (zeros
strictly above the diagonal), and rows summing to 1.

**You'll know it clicked when**: you can explain, without looking at the
code, why `trick_v2`'s row-normalized lower-triangular matrix and a real
attention head's `wei` matrix have the *same shape and same constraints*
(lower-triangular, rows sum to 1) but different *values* — and why the
`* head_size**-0.5` scaling is there (what breaks if you remove it and
`head_size` is large).
