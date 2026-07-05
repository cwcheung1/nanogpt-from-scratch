# 04 — Multi-Head Attention

**Concept**: run several small attention heads in parallel instead of one
big one, concatenate their outputs, and project back down — same total
compute budget, but able to represent several distinct kinds of
relationships (e.g. "what's the previous vowel," "what character started
this word") at once instead of one averaged-together pattern. This lesson
also plugs attention into a real trainable model for the first time.

**Analogy**: one attention head is one person in the discussion from lesson
3's analogy, asking one kind of question. Multi-head is convening several
such people simultaneously, each with a different question in mind, then
having a moderator (the output projection) synthesize their separate
findings into one summary.

**How it works** (`lessons/code/04_multi_head_attention.py`):
- `MultiHeadAttention` is literally `nn.ModuleList([Head(head_size) for _ in
  range(num_heads)])`, run independently and `torch.cat`'d along the last
  dimension, then passed through one more `nn.Linear` (`proj`). Note
  `head_size = n_embd // n_head` — splitting one 32-wide embedding into 4
  heads of size 8 keeps the concatenated output back at width 32, so this
  costs no extra parameters over one 32-wide head, it just restructures the
  same budget into 4 independent "conversations."
- **Position embeddings, introduced here for the first time**: attention has
  no built-in sense of sequence order — `q @ k.T` treats the input as a *set*
  of tokens, not a *sequence*. `self.position_embedding_table(torch.arange(T))`
  gives each position 0..T-1 its own learned vector, added directly to the
  token embedding (`x = tok_emb + pos_emb`). This is the mechanism that lets
  the model tell "cat sat" from "sat cat" apart. (GPT-2/nanoGPT use this
  same additive learned-position-embedding scheme; rotary/ALiBi are
  alternatives you'll meet in production models, not covered here.)
- `generate` now slices `idx[:, -block_size:]` before each forward pass —
  the bigram model in lesson 2 could ignore this because it only ever used
  the last token regardless of how much history you fed it, but an
  attention model's `position_embedding_table` only has `block_size` rows,
  so the input window must be capped. This is your first encounter with
  *context window* as a hard architectural limit, not just a training
  convenience.
- Everything else (train loop, `estimate_loss`, cross-entropy reshaping) is
  identical to lesson 2 — the point of this lesson is isolating "what did
  adding attention change," not introducing new training mechanics.

**Look at**: `lessons/code/04_multi_head_attention.py`

**Run this**:
```
make lesson4
```
Expect: ~8.6k params (vs. bigram's 4.2k) and val loss settling around
**2.33-2.36** — a real, measurable improvement over lesson 2's ~2.48-2.50
plateau, from the same dataset and comparable training budget. The sampled
text is still gibberish, but look for slightly better local structure
(more plausible short "words") than lesson 2's output.

**You'll know it clicked when**: you can explain *why* removing the
position embedding would hurt this model specifically (what information
does attention alone have access to?) — and you can predict, before
running it, whether increasing `n_head` from 4 to 8 while holding `n_embd`
at 32 changes the model's total parameter count (it shouldn't, much —
`head_size` just shrinks to keep the concatenated width the same).
