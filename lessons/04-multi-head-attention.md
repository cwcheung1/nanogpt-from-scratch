# 04 — Multi-Head Attention

*Before this lesson: [00 — Roadmap](00-roadmap.md) and lesson 3 (this
reuses lesson 3's `Head` unchanged — if query/key/value or the causal mask
are still fuzzy, that's the lesson to revisit, not this one).*

**Jargon buster — new terms this lesson's code uses**:

- **`nn.ModuleList([...])`** — a Python list of layers that PyTorch still
  recognizes as part of the model (a plain Python list wouldn't register
  its contents' parameters) — holds the independent attention heads.

- **`torch.cat([...], dim=-1)`** — glues several tensors together along the
  given dimension; here, concatenates each head's output side by side.

- **`torch.arange(T)`** — just `[0, 1, 2, ..., T-1]` as a tensor — used to
  look up "position 0's vector, position 1's vector, ..." in the position
  embedding table.

**Concept**: run several small attention heads in parallel instead of one
big one, concatenate their outputs, and project back down — same total
compute budget, but able to represent several distinct kinds of
relationships (e.g. "what's the previous vowel," "what character started
this word") at once instead of one averaged-together pattern. This lesson
also plugs attention into a real trainable model for the first time — the
first time in this series you'll see attention's val loss number, and the
first real comparison point against lesson 2's bigram score.

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
  no built-in sense of sequence order. Concretely: the dot product `q @ k.T`
  that scores "how relevant is position j to position i" only looks at the
  *content* of each position's vector — it has no idea whether j came right
  before i or 5 characters earlier, only that it comes at-or-before i (via
  the causal mask). Feed it the characters of "cat sat" vs. "sat cat" *in
  whatever order, unlabeled* and attention alone can't tell them apart — it
  only sees "a `c`, an `a`, a `t`, a ` `, an `s`... available to attend to,"
  not their positions. Fix: `self.position_embedding_table(torch.arange(T))`
  gives each position 0..T-1 its own learned vector purely for "this is
  where I sit in the sequence," added directly to the token embedding
  (`x = tok_emb + pos_emb`) so every position's vector now encodes *both*
  "which character am I" and "where am I." (GPT-2/nanoGPT use this same
  additive learned-position-embedding scheme; rotary/ALiBi are alternatives
  you'll meet in production models, not covered here.)

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
