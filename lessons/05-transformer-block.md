# 05 — The Transformer Block

**Concept**: a transformer block is "communicate, then compute" —
multi-head attention lets tokens exchange information, a per-position MLP
lets the model process what it gathered — and both are wrapped in residual
("skip") connections and preceded by LayerNorm so you can stack many of
these blocks without training collapsing.

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
    transformation and gets added back. This gives gradients a direct path
    backward through addition (`d/dx (x + f(x)) = 1 + f'(x)`) instead of
    having to flow through every block's full transformation with nothing
    to fall back on.
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
