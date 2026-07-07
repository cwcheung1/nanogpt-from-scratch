# 05 — The Transformer Block

*Before this lesson: [00 — Roadmap](00-roadmap.md) and lesson 4 — this
reuses lesson 4's `Head`/`MultiHeadAttention` unchanged, and adds 3 new
ideas on top: a per-position MLP, residual connections, and LayerNorm, all
defined in plain language below before they're used.*

**Jargon buster — new terms this lesson's code uses**:

- **`nn.Sequential(layer1, layer2, ...)`** — runs its input through each
  layer in order, output of one feeding into the next; used for `FeedForward`'s
  Linear→ReLU→Linear and for stacking the `Block`s themselves.

- **`nn.ReLU()`** — a simple nonlinearity: replace every negative number
  with 0, leave positive numbers unchanged. Without *some* nonlinearity
  between the two `Linear` layers, stacking them would collapse
  mathematically into one bigger linear layer — no added expressive power.

- **`nn.LayerNorm(n_embd)`** — see "Two new terms" below in this lesson's
  main text; it's defined there in full rather than repeated here.

**Concept**: a transformer block is "communicate, then compute" —
multi-head attention (lesson 4, unchanged) lets positions exchange
information with each other, then a per-position MLP ("multi-layer
perceptron" — just a small stack of learned layers, here 2 `nn.Linear`
layers with a nonlinearity between them) lets the model actually process
what it just gathered. Both steps are wrapped in a **residual ("skip")
connection** and preceded by **LayerNorm**, two stability tricks defined
below that make it possible to stack many of these blocks deep without
training falling apart.

**Two new terms, defined before you hit them in the code:**

- **Residual / skip connection**: instead of `x = f(x)` (replace the input
  entirely with the transformation's output), do `x = x + f(x)` (add the
  transformation's output ON TOP of the original input). The original
  information always survives untouched, in addition to whatever new thing
  this step computed — nothing is ever fully overwritten. Why this matters
  for training specifically is explained below, and demonstrated with a
  real side-by-side experiment, not just asserted.

- **LayerNorm**: a fixed (non-learned-in-the-interesting-sense) rescaling
  step, applied to every position's vector independently, that shifts and
  scales its numbers to have a consistent mean (~0) and spread (~1) before
  they're fed into the next computation. Think of it as "regardless of how
  large or small the numbers flowing through the network have gotten by
  this point, renormalize them to a consistent, well-behaved range before
  the next step" — a stability measure, not a source of new information.

**Analogy**: think of each block as one round of a group work session:
first everyone shares relevant info with each other (attention), then
everyone goes off individually to think about what they just heard
(feedforward). Residual connections are the rule "you always keep your
original notes, in addition to whatever new insight this round produced" —
without that rule, information has to survive being *fully rewritten* at
every single round, and the more rounds you stack, the more likely it gets
mangled before it reaches the end.

**How it works** (`lessons/code/05_transformer_block.py`):

- `FeedForward`: `Linear(n_embd, 4*n_embd) -> ReLU -> Linear(4*n_embd,
  n_embd)`. The 4x expansion-then-contraction matches GPT-2 and the
  original Transformer paper — attention moved information *between*
  positions, this is where per-position "thinking" capacity actually lives.

- `Block.forward`: `x = x + self.sa(self.ln1(x))` then `x = x +
  self.ffwd(self.ln2(x))`. Two things to notice:

  - **Pre-norm**: `LayerNorm` is applied *before* attention/feedforward, not
    after. The original 2017 "Attention Is All You Need" paper used
    post-norm; GPT-2 switched to pre-norm because it trains more stably at
    depth, and every modern LLM (including, as far as public architecture
    descriptions go, Claude) follows suit.

  - **Residual = `x + f(x)`, not `x = f(x)`**: the input skips around the
    transformation and gets added back. Recall from the roadmap glossary
    that training's "backward pass" works out, for every parameter, "which
    direction would reduce the loss" by working backward through the
    network one step at a time. With plain `x = f(x)` stacked 4 (or 40)
    deep, that backward signal has to pass through every single
    transformation in the stack, and can shrink toward nothing or blow up
    along the way — the deeper the stack, the worse this gets. Addition
    (`x + f(x)`) gives that backward signal a second, untransformed path
    straight through every block, in parallel with the path through `f(x)`
    — so even if the `f(x)` path degrades, the signal still gets through via
    the `+x` path. (For the calculus-inclined: `d/dx (x + f(x)) = 1 + f'(x)`
    — that `+1` is exactly the guaranteed direct path. You don't need to be
    able to derive this to use it; the ablation below shows the effect
    directly, without any calculus.)

- `GPTLanguageModel` stacks `n_layer=4` blocks via `nn.Sequential`, then one
  final `ln_f` before the output head — same token+position embedding setup
  as lesson 4.

- **The ablation** (`use_residual=False` path in `Block.forward`): identical
  architecture, identical seed, identical hyperparameters — the *only*
  difference is `x = self.sa(self.ln1(x))` (replace) instead of `x =
  x + self.sa(self.ln1(x))` (add). This isolates residual connections as the
  single variable, which is the point: don't take "residuals help" on
  faith, watch it fail without them.

**Look at**: `lessons/code/05_transformer_block.py`

**Run this**:
```
make lesson5
```
Actual results from this run (yours will match closely, same seed):
| | val loss @ step 0 | val loss @ step 2999 |
|---|---|---|
| WITH residuals | 4.22 | **2.10** |
| WITHOUT residuals | 4.48 | **3.34** (noisy, non-monotonic) |

WITH residuals clearly beats lesson 4's single-attention-layer result
(~2.33-2.36) — depth is now paying off. WITHOUT residuals is *worse than
lesson 2's bigram baseline* (~2.48) despite having 13x the parameters and a
much more expressive architecture, and its loss bounces around late in
training instead of monotonically improving — this is what "gradients can't
propagate through depth" looks like empirically, not just in theory.

**You'll know it clicked when**: you can explain why `d/dx(x + f(x))`
containing a `+1` term matters for training a 4-layer (or 40-layer) network,
and why removing residuals hurts *more* as `n_layer` increases (try bumping
`n_layer` to 8 in both variants and predict which gap widens more before you
run it).
